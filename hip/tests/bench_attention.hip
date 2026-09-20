// k_window_attention across every width the network runs, and what it costs per frame.
//
// It is the only one of the five hot kernels that has had nothing done to it: Q, K and V are
// activations, so no weight-format work ever reached them, and the per-block profile cannot isolate
// it because it runs once in all 71 blocks. bench-block0 puts it at 2.673 ms in block 0 alone --
// the largest kernel there -- but block 0 is the widest case and extrapolating from it would be the
// stale-profile mistake again. So measure each group and add them up.
//
// The block geometry is network.hip's: five channel widths, and shift[] gives each block a 0 or 4
// pixel pad that rounds sw/sh up to a multiple of 8. This uses the unpadded size for each group,
// which understates by ~1-2% on the shifted blocks; the counts are exact.
//
//   usage: ./bench-attention [iters]
#include "dlss5_launch.hpp"
#include "dlss5_runtime.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace dlss5;

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 20;
    hip_check(hipSetDevice(0), "device");
    hipDeviceProp_t prop{};
    hip_check(hipGetDeviceProperties(&prop, 0), "props");
    std::printf("bench-attention device=%s arch=%s qkv=%s iters=%d\n\n", prop.name,
                prop.gcnArchName, DLSS5_QKV16 ? "f16" : "e4m3", iters);

    struct Group { const char* what; uint c, w, h, count; };
    // blocks 0 and 70; 1-4 and 66-69; 5-8 and 62-65; 9-14 and 56-61; 15-22 and 48-55; 23-30 and 40-47
    const Group groups[] = {
        {"C32  1920x1152  (blocks 0, 70)",        32, 1920, 1152,  2},
        {"C32   960x576   (1-4, 66-69)",          32,  960,  576,  8},
        {"C64   480x288   (5-8, 62-65)",          64,  480,  288,  8},
        {"C128  240x144   (9-14, 56-61)",        128,  240,  144, 12},
        {"C256  120x72    (15-22, 48-55)",       256,  120,   72, 16},
        {"C512   60x36    (23-30, 40-47)",       512,   60,   36, 16},
    };

    hipStream_t stream;
    hip_check(hipStreamCreate(&stream), "stream");
    set_launch_stream(stream);
    hipEvent_t e0, e1;
    hip_check(hipEventCreate(&e0), "ev"); hip_check(hipEventCreate(&e1), "ev");

    // One allocation, sized for the largest case, reused by every group.
    const size_t nmax = 1928ull * 1160;
    // The qkv buffer doubles under DLSS5_QKV16; the seed fill below covers whichever size.
    DeviceBuf qkv(nmax * 3 * 32 * (DLSS5_QKV16 ? 2 : 1)), out(nmax * 32),
        bias(16ull * 4096 * 4);
    {
        std::vector<u8> seed(1u << 20);
        for (size_t i = 0; i < seed.size(); ++i) seed[i] = u8((i * 131u + i / 97u) & 0x7fu);
        for (size_t off = 0; off + seed.size() <= qkv.bytes; off += seed.size())
            hip_check(hipMemcpy((char*)qkv.p + off, seed.data(), seed.size(),
                                hipMemcpyHostToDevice), "seed");
        hip_check(hipMemset(bias.p, 0, bias.bytes), "bias");
    }

    double frame = 0;
    std::printf("%-34s %10s %8s %12s\n", "group", "ms/call", "calls", "ms/frame");
    for (const Group& g : groups) {
        // The C512 buffers here are 16 heads x 4096 bias floats; the allocation covers the largest.
        auto go = [&] {
            launch_attention(int(g.c), qkv.as<u8>(), bias.as<float>(), out.as<u8>(), g.w, g.h, 1, true);
        };
        for (int i = 0; i < 3; ++i) go();
        hip_check(hipStreamSynchronize(stream), "warm");
        hip_check(hipEventRecord(e0, stream), "e0");
        for (int i = 0; i < iters; ++i) go();
        hip_check(hipEventRecord(e1, stream), "e1");
        hip_check(hipEventSynchronize(e1), "sync");
        float ms = 0; hip_check(hipEventElapsedTime(&ms, e0, e1), "elapsed");
        ms /= iters;
        double per_frame = double(ms) * g.count;
        frame += per_frame;
        std::printf("%-34s %10.3f %8u %12.3f\n", g.what, ms, g.count, per_frame);
    }
    std::printf("\n  k_window_attention per frame: %.2f ms\n", frame);
    std::printf("  against a 98 ms model that is %.0f%%.\n", 100.0 * frame / 98.0);
    std::printf("\n  (bench-block0 measures the C32 1920x1152 case at 2.673 ms -- if the first row\n"
                "   above disagrees with that, one of the two benches is wrong.)\n");
    return 0;
}
