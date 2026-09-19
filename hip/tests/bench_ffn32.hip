// Isolated k_ffn_f32<32> microbenchmark: real C32 stage size (960x576 tokens),
// precise_c32 activation path. Reports ms/call and an output hash so source
// variants can be A/B checked for bit-identity without the full graph suite.
#include "dlss5_launch.hpp"
#include "dlss5_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
using namespace dlss5;
int main(int argc, char** argv) {
    const int iters = argc > 2 ? std::atoi(argv[2]) : 20;
    const uint c = argc > 3 ? std::stoul(argv[3]) : 32u;

    if (c != 32 && c != 64 && c != 128 && c != 256) {
        std::printf("C must be 32, 64, 128 or 256\n");
        return 2;
    }

    // The network quarters the token count each time it doubles C, so each stage does comparable
    // arithmetic. Matching that here keeps the four measurements on the same footing instead of making
    // the wide ones look expensive purely for covering more tokens. "-" for argv[1] takes this default.
    const uint defaults = 552960u / ((c / 32u) * (c / 32u));
    const bool explicit_tokens = argc > 1 && std::string(argv[1]) != "-";
    const uint tokens = explicit_tokens ? std::stoul(argv[1]) : defaults;
    std::vector<float> x(size_t(tokens) * c);
    for (size_t i = 0; i < x.size(); ++i) x[i] = H(float(int((i * 2654435761u) % 1000) - 500) * 0.002f);
    std::vector<float> w(8 * c * c);
    for (size_t i = 0; i < w.size(); ++i) w[i] = float(int((i * 40503u + i / 37u) % 97) - 48) * 0.01f;
    std::vector<float> sc(c);
    for (uint i = 0; i < c; ++i) sc[i] = H(float(int(i % 13) - 6) * 0.077f);
    std::vector<u8> tiled(w.size());
    pack_tiled_e4m3(tiled.data(), w.data(), 4 * c, c);
    pack_tiled_e4m3(tiled.data() + 4 * c * c, w.data() + 4 * c * c, c, 4 * c);

    // The same weights, stored as f16 in the same tile layout. Deliberately round-tripped through E4M3
    // first: the matrix unit must see values identical to the E4M3 run, or the timing comparison is
    // measuring two different computations and means nothing. The output hashes below prove it.
    auto pack_tiled_f16 = [](u8* dst, const float* src, size_t N, size_t K) {
        __half* d = reinterpret_cast<__half*>(dst);
        for (size_t t = 0; t < N / 16; t++)
            for (size_t g = 0; g < K / 32; g++)
                for (size_t k = 0; k < 32; k++)
                    for (size_t j = 0; j < 16; j++)
                        d[(t * (K / 32) + g) * 512 + k * 16 + j] =
                            __float2half(from_e4m3(e4m3_byte(src[(t * 16 + j) * K + g * 32 + k])));
    };

    std::vector<u8> tiled16(w.size() * 2);
    pack_tiled_f16(tiled16.data(), w.data(), 4 * c, c);
    pack_tiled_f16(tiled16.data() + 2 * 4 * c * c, w.data() + 4 * c * c, c, 4 * c);
    DeviceBuf dx(x.size() * 4), dw(tiled.size()), ds(sc.size() * 4), dy(size_t(tokens) * c * 4);
    dx.upload(x.data(), x.size() * 4); dw.upload(tiled.data(), tiled.size());
    ds.upload(sc.data(), sc.size() * 4);
    hipStream_t stream; hip_check(hipStreamCreate(&stream), "stream");
    set_launch_stream(stream);
    auto go = [&] {
        launch_ffn_f32(c, dx.as<float>(), dw.as<u8>(), ds.as<float>(), dy.as<float>(), tokens, false,
                       c == 32);
    };
    for (int i = 0; i < 5; ++i) go();
    hip_check(hipStreamSynchronize(stream), "warm");
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0, stream);
    for (int i = 0; i < iters; ++i) go();
    hipEventRecord(e1, stream);
    hip_check(hipEventSynchronize(e1), "sync");
    float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= iters;
    std::printf("FFN C=%u tokens=%u iters=%d: %.3f ms/call\n", c, tokens, iters, ms);
    std::vector<float> out(size_t(tokens) * c);
    dy.download(out.data(), out.size() * 4);
    uint64_t hsh = 1469598103934665603ull;
    for (float v : out) { hsh ^= uint64_t(as_u32(v)); hsh *= 1099511628211ull; }
    std::printf("out_hash=%016llx min=%g max=%g\n", (unsigned long long)hsh,
                *std::min_element(out.begin(), out.end()), *std::max_element(out.begin(), out.end()));

    // --- the same kernel, the same numbers, f16 weights ------------------------------------------
    //
    // RDNA3 has no FP8 matrix instruction, so an E4M3 weight fragment is filled by load_e4m3(): eight
    // strided single-byte reads and eight conversions per lane. The f16 path is one load_matrix_sync.
    // The difference between these two timings is what that costs, and it is the only number that says
    // whether storing the weights wide is worth 148 MB more VRAM.
    DeviceBuf dw16(tiled16.size()), dy16(size_t(tokens) * c * 4);
    dw16.upload(tiled16.data(), tiled16.size());

    auto go16 = [&] {
        launch_ffn_f32_w16(c, dx.as<float>(), dw16.as<u8>(), ds.as<float>(), dy16.as<float>(), tokens,
                           false, c == 32);
    };

    for (int i = 0; i < 5; ++i) go16();
    hip_check(hipStreamSynchronize(stream), "warm16");
    hipEventRecord(e0, stream);
    for (int i = 0; i < iters; ++i) go16();
    hipEventRecord(e1, stream);
    hip_check(hipEventSynchronize(e1), "sync16");
    float ms16 = 0; hipEventElapsedTime(&ms16, e0, e1); ms16 /= iters;

    std::vector<float> out16(size_t(tokens) * c);
    dy16.download(out16.data(), out16.size() * 4);
    uint64_t hsh16 = 1469598103934665603ull;
    for (float v : out16) { hsh16 ^= uint64_t(as_u32(v)); hsh16 *= 1099511628211ull; }

    std::printf("FFN C=%u f16-weights: %.3f ms/call  out_hash=%016llx\n", c, ms16,
                (unsigned long long)hsh16);

    if (hsh16 != hsh) {
        // Without this the timing above would be compared against a different computation, which is
        // exactly the sort of confident nonsense this whole exercise is meant to avoid.
        std::printf("MISMATCH: the f16 run computed something different, so the timing is meaningless. "
                    "Ignore the comparison.\n");
        return 1;
    }

    std::printf("identical output. E4M3 %.3f ms -> f16 %.3f ms = %+.1f%% (%.3f ms)\n", ms, ms16,
                (ms16 - ms) / ms * 100.0, ms16 - ms);
    // The full 2x2. Every cell must produce the same hash -- the f16 weights are the dequantised E4M3
    // values and the f16 hidden buffer holds the E4M3-rounded ones, so all four compute the same thing.
    // A16 doubles LDS, so if it helps alone and hurts beside W16, occupancy is the reason.
    {
        std::printf("\n%-28s%10s  %s\n", "weights / hidden", "ms/call", "out_hash");

        struct Cell { const char* name; bool w16, a16; };
        const Cell cells[] = { { "E4M3    / E4M3", false, false },
                               { "f16     / E4M3", true, false },
                               { "E4M3    / f16", false, true },
                               { "f16     / f16", true, true } };
        float first_ms = 0.f;

        for (const Cell& cell : cells) {
            DeviceBuf dyc(size_t(tokens) * c * 4);
            const u8* wsel = cell.w16 ? dw16.as<u8>() : dw.as<u8>();

            auto run = [&] {
                launch_ffn_f32_bench(c, dx.as<float>(), wsel, ds.as<float>(), dyc.as<float>(), tokens,
                                     false, c == 32, cell.w16, cell.a16);
            };

            for (int i = 0; i < 5; ++i) run();
            hip_check(hipStreamSynchronize(stream), "warm cell");
            hipEventRecord(e0, stream);
            for (int i = 0; i < iters; ++i) run();
            hipEventRecord(e1, stream);
            hip_check(hipEventSynchronize(e1), "sync cell");
            float cms = 0; hipEventElapsedTime(&cms, e0, e1); cms /= iters;

            std::vector<float> co(size_t(tokens) * c);
            dyc.download(co.data(), co.size() * 4);
            uint64_t ch = 1469598103934665603ull;
            for (float v : co) { ch ^= uint64_t(as_u32(v)); ch *= 1099511628211ull; }

            if (first_ms == 0.f) first_ms = cms;
            std::printf("%-28s%10.3f  %016llx%s%s\n", cell.name, cms, (unsigned long long)ch,
                        ch == hsh ? "" : "   HASH MISMATCH -- timing meaningless",
                        cms < first_ms ? "" : "");
        }

        std::printf("%-28s%10s  (relative to E4M3/E4M3)\n", "", "");
    }

    std::printf("C=%u: %s\n", c,
                ms16 < ms * 0.9 ? "the E4M3 load path costs real time at this width"
                : ms16 > ms * 1.05
                    ? "f16 weights are SLOWER at this width -- doubled weight traffic outweighs the "
                      "cheaper load, so a network-wide conversion would have to be per-C"
                    : "no meaningful difference at this width; the loader is not the limit here");
    std::printf("weights=%zu KiB as E4M3, %zu KiB as f16\n", size_t(8 * c * c) / 1024,
                size_t(2 * 8 * c * c) / 1024);
    return 0;
}
