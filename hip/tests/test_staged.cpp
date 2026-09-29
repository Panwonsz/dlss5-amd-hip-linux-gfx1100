// dlss5_run_staged against the daemon's own code (2026-09-29).
//
// For every format the daemon accepts: a 1920x1080 frame with padded rows goes through
//   reference: the daemon's DecodeRow -> dlss5_run -> EncodeRow (its very source, included below)
//   staged:    dlss5_run_staged (conversion on the GPU)
// with the same seed, and the two answers must be equal byte for byte, row padding included. A frame
// with a NaN in it must fail on both paths.
//
//   make -C hip ARCH=gfx1100 libdlss5_hip.so test-staged
//   hip/build/test_staged <weights dir> [path/to/libdlss5_hip.so]
#define main dlss5_daemon_main
#include "../../linux/dlss5_nr_daemon.cpp"
#undef main

#include <dlfcn.h>
#include <random>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <weights dir> [libdlss5_hip.so]\n", argv[0]);
        return 2;
    }
    const char* lib = argc > 2 ? argv[2] : "hip/libdlss5_hip.so";
    void* h = dlopen(lib, RTLD_NOW);
    if (!h) {
        std::fprintf(stderr, "dlopen %s: %s\n", lib, dlerror());
        return 2;
    }
    auto init = (int (*)(const char*, int))dlsym(h, "dlss5_init");
    auto run = (int (*)(const float*, float*, unsigned))dlsym(h, "dlss5_run");
    auto staged = (int (*)(const void*, void*, unsigned, unsigned, unsigned))dlsym(h, "dlss5_run_staged");
    auto error = (const char* (*)())dlsym(h, "dlss5_last_error");
    if (!init || !run || !staged || !error) {
        std::fprintf(stderr, "%s lacks dlss5_run_staged (rebuild it)\n", lib);
        return 2;
    }
    if (init(argv[1], 0) != 0) {
        std::fprintf(stderr, "init: %s\n", error());
        return 2;
    }

    const uint32_t formats[] = {kR8G8B8A8_UNORM, kB8G8R8A8_UNORM_SRGB, kR16G16B16A16_FLOAT,
                                kR32G32B32A32_FLOAT, kR10G10B10A2_UNORM, kR11G11B10_FLOAT,
                                kR8G8B8A8_UNORM_SRGB, kB8G8R8A8_UNORM};
    const uint32_t W = kModelWidth, H = kModelHeight;
    std::vector<float> input(size_t(W) * H * 4), answer(size_t(W) * H * 3);
    std::mt19937 rng(20260929);
    std::uniform_real_distribution<float> unit(0.0f, 1.25f);
    int fails = 0;
    unsigned seed = 11;
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    for (uint32_t format : formats) {
        const uint32_t bpp = format == kR32G32B32A32_FLOAT ? 16 : format == kR16G16B16A16_FLOAT ? 8 : 4;
        const uint32_t pitch = (W * bpp + 64 + 255) / 256 * 256; // padded like a D3D12 footprint, plus some
        const size_t bytes = size_t(pitch) * H;
        std::vector<uint8_t> in(bytes);
        for (auto& b : in) b = uint8_t(rng());
        for (uint32_t y = 0; y < H; y++) {
            uint8_t* row = in.data() + size_t(y) * pitch;
            for (uint32_t x = 0; x < W; x++) {
                if (format == kR16G16B16A16_FLOAT)
                    for (int c = 0; c < 4; c++) reinterpret_cast<uint16_t*>(row)[x * 4 + c] = FloatToHalf(unit(rng));
                else if (format == kR32G32B32A32_FLOAT)
                    for (int c = 0; c < 4; c++) reinterpret_cast<float*>(row)[x * 4 + c] = unit(rng);
            }
        }

        // The daemon: decode, model, encode over a copy of the request (so the padding comes back as sent).
        std::vector<uint8_t> reference = in;
        const auto t0 = clk::now();
        for (uint32_t y = 0; y < H; y++)
            DecodeRow(in.data() + size_t(y) * pitch, input.data() + size_t(y) * W * 4, format, W);
        const int rc = run(input.data(), answer.data(), seed);
        for (uint32_t y = 0; y < H && rc == 0; y++)
            EncodeRow(answer.data() + size_t(y) * W * 3, reference.data() + size_t(y) * pitch, format, W);
        const auto t1 = clk::now();

        std::vector<uint8_t> out(bytes, 0xCD);
        const int rs = staged(in.data(), out.data(), format, pitch, seed);
        const auto t2 = clk::now();

        if (rc != 0 || rs != 0) {
            std::printf("format %3u: FAIL  daemon path rc=%d, staged rc=%d (%s)\n", format, rc, rs, error());
            fails++;
            continue;
        }
        size_t bad = 0, first = 0;
        for (size_t i = 0; i < bytes; i++)
            if (out[i] != reference[i] && bad++ == 0) first = i;
        if (bad) {
            std::printf("format %3u: FAIL  %zu of %zu bytes differ, first at %zu (row %zu, byte %zu): daemon %02x staged %02x\n",
                        format, bad, bytes, first, first / pitch, first % pitch, reference[first], out[first]);
            fails++;
        } else {
            std::printf("format %3u: byte-exact (%zu bytes, pitch %u)  daemon path %.1f ms, staged %.1f ms\n",
                        format, bytes, pitch, ms(t0, t1), ms(t1, t2));
        }
        seed++;
    }

    // A NaN in the frame: both paths must refuse it.
    {
        const uint32_t pitch = W * 8;
        std::vector<uint8_t> in(size_t(pitch) * H, 0), out(in.size());
        reinterpret_cast<uint16_t*>(in.data())[1000 * 4 + 1] = 0x7E00; // half NaN, green of pixel 1000
        for (uint32_t y = 0; y < H; y++)
            DecodeRow(in.data() + size_t(y) * pitch, input.data() + size_t(y) * W * 4, kR16G16B16A16_FLOAT, W);
        const int rc = run(input.data(), answer.data(), seed);
        const int rs = staged(in.data(), out.data(), kR16G16B16A16_FLOAT, pitch, seed);
        if (rc == 0 || rs == 0) {
            std::printf("NaN frame: FAIL  daemon path rc=%d, staged rc=%d\n", rc, rs);
            fails++;
        } else {
            std::printf("NaN frame: refused by both (%s)\n", error());
        }
    }

    std::printf("dlss5_run_staged vs daemon: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
