// What block 0's 12.6 ms is actually made of.
//
// The per-block profile (DLSS5_HIP_TIME=1) times from a host clock taken BEFORE the 33 MB input
// upload, while gpu_ms starts after it -- so block 00's figure includes a PCIe transfer that the
// 97.9 ms headline does not. This reproduces the whole interval at real sizes and times each piece,
// including that upload from both pageable and pinned host memory.
//
// The weights are synthetic: this measures time, not numbers, and every kernel here has its own
// correctness cover elsewhere (test_linear, test_ffn_f32, test_attention, hip-network70's mean).
//
// The self-check is the sum. If it does not land near the 12.6 ms the real profile reports, the
// reconstruction is wrong and none of the per-kernel figures mean anything -- so the sum is printed
// against that number rather than left for the reader to do.
//
//   usage: ./bench-block0 [iters]
#include "dlss5_launch.hpp"
#include "dlss5_runtime.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
using namespace dlss5;

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 20;
    const uint W = 1920, Ht = 1152, C = 32;
    const uint sw = W, sh = Ht;              // block 0: shift=0, so no padding
    const uint n = sw * sh;                  // 2,211,840 tokens
    const size_t nsc = size_t(W + 8) * (Ht + 8) * 32;   // Scratch's own sizing
    const size_t rgb_bytes = 1920ull * 1080 * 16;

    hip_check(hipSetDevice(0), "device");
    hipDeviceProp_t prop{};
    hip_check(hipGetDeviceProperties(&prop, 0), "props");
    // The formats come from the LINKED kernels, not from this file's macros -- a stale
    // build/kernels.o once made this line assert e4m3 while the kernels wrote f16, and the
    // mismatched buffer size page-faulted three rows later.
    std::printf("bench-block0 device=%s arch=%s weights=%s kernels[%s] tokens=%u iters=%d\n",
                prop.name, prop.gcnArchName, dlss5_w16() ? "f16" : "e4m3",
                dlss5_kernel_formats(), n, iters);

    // ---- weights, in whichever format the kernels will read them ------------------------------
    auto fill = [](std::vector<float>& v, unsigned salt) {
        for (size_t i = 0; i < v.size(); ++i)
            v[i] = float(int((i * 7919u + salt) % 17) - 8) * 0.03125f;
    };
    const size_t m = size_t(C) * C;
    std::vector<float> ffn(8 * m), qkv(3 * m), proj(m);
    fill(ffn, 1); fill(qkv, 2); fill(proj, 3);

    // The FFN's two halves share one buffer, expand then contract -- one byte an element as E4M3,
    // two as f16, exactly as network.hip packs it.
    DeviceBuf fw;
    if (dlss5_w16()) {
        std::vector<__half> t(8 * m);
        pack_tiled_half(t.data(), ffn.data(), 4 * C, C);
        pack_tiled_half(t.data() + 4 * m, ffn.data() + 4 * m, C, 4 * C);
        fw.alloc(t.size() * 2); fw.upload(t.data(), t.size() * 2);
    } else {
        std::vector<u8> t(8 * m);
        pack_tiled_e4m3(t.data(), ffn.data(), 4 * C, C);
        pack_tiled_e4m3(t.data() + 4 * m, ffn.data() + 4 * m, C, 4 * C);
        fw.alloc(t.size()); fw.upload(t.data(), t.size());
    }
    DeviceBuf qw = upload_tiled_weights(qkv.data(), 3 * C, C);
    DeviceBuf p1 = upload_tiled_weights(proj.data(), C, C);

    auto floats_of = [](size_t count, float base) {
        std::vector<float> v(count);
        for (size_t i = 0; i < count; ++i) v[i] = base + float(i % 7) * 0.01f;
        DeviceBuf b(count * 4); b.upload(v.data(), count * 4); return b;
    };
    DeviceBuf fs = floats_of(C, 0.9f), qs = floats_of(C / 32, 1.0f);
    DeviceBuf bias = floats_of((C / 32) * 4096, 0.f), s1 = floats_of(C, 0.9f);
    DeviceBuf prefix_w = floats_of(512, 0.5f);

    // ---- buffers -------------------------------------------------------------------------------
    DeviceBuf rgb(rgb_bytes), color(size_t(n) * 16), tiles(size_t(n) * 16);
    // q8 doubles under DLSS5_QKV16, exactly as Scratch::q8 does.
    DeviceBuf a(nsc * 4), a16(nsc * 2), bh(nsc * 2), q8(nsc * 3 * (DLSS5_QKV16 ? 2 : 1)),
        att8(nsc), ah(nsc * 2);
    DeviceBuf skip0(size_t(n) * 32 * 4);

    std::vector<float> host_pageable(rgb_bytes / 4, 0.5f);
    void* host_pinned = nullptr;
    const bool have_pinned = hipHostMalloc(&host_pinned, rgb_bytes) == hipSuccess;
    if (have_pinned) ::memcpy(host_pinned, host_pageable.data(), rgb_bytes);

    hipStream_t stream;
    hip_check(hipStreamCreate(&stream), "stream");
    set_launch_stream(stream);
    hipEvent_t e0, e1;
    hip_check(hipEventCreate(&e0), "ev"); hip_check(hipEventCreate(&e1), "ev");

    double total = 0;
    auto time_it = [&](const char* name, auto&& fn) {
        for (int i = 0; i < 3; ++i) fn();
        hip_check(hipStreamSynchronize(stream), "warm");
        hip_check(hipEventRecord(e0, stream), "e0");
        for (int i = 0; i < iters; ++i) fn();
        hip_check(hipEventRecord(e1, stream), "e1");
        hip_check(hipEventSynchronize(e1), "sync");
        float ms = 0; hip_check(hipEventElapsedTime(&ms, e0, e1), "elapsed");
        ms /= iters; total += ms;
        std::printf("  %-34s %7.3f ms\n", name, ms);
        return ms;
    };

    std::printf("\nthe 33 MB input upload (outside gpu_ms, inside block 00's figure):\n");
    double pageable = time_it("hipMemcpyAsync H2D pageable", [&] {
        hipMemcpyAsync(rgb.p, host_pageable.data(), rgb_bytes, hipMemcpyHostToDevice, stream);
    });
    double pinned = 0;
    if (have_pinned)
        pinned = time_it("hipMemcpyAsync H2D pinned", [&] {
            hipMemcpyAsync(rgb.p, host_pinned, rgb_bytes, hipMemcpyHostToDevice, stream);
        });
    total -= pinned;   // only one of the two is in the real path

    std::printf("\nkernels, in the order block 0 runs them:\n");
    time_it("launch_rgb_graph", [&] {
        launch_rgb_graph(rgb.as<float>(), color.as<float>(), tiles.as<float>(), W, 1080, W, Ht);
    });
    time_it("launch_prefix_h", [&] {
        launch_prefix_h(tiles.as<float>(), prefix_w.as<float>(), nullptr, a.as<float>(),
                        a16.as<__half>(), 0u, W, Ht, 0u);
    });
    // window(), C32 path, in_h non-null: the FFN takes its f16 input and emits f16.
    time_it("launch_ffn_f32_hin_hout", [&] {
        launch_ffn_f32_hin_hout(32, a16.as<__half>(), fw.as<u8>(), fs.as<float>(),
                                bh.as<__half>(), n, false, true);
    });
    time_it("launch_qkv_norm_f32_hin", [&] {
        launch_qkv_norm_f32_hin(32, bh.as<__half>(), qw.as<u8>(), q8.as<u8>(), n,
                                qs.as<float>(), 1.f, true);
    });
    time_it("launch_attention", [&] {
        launch_attention(32, q8.as<u8>(), bias.as<float>(), att8.as<u8>(), sw, sh, 1, true);
    });
    time_it("launch_linear_f32_f8in_hrid_hout", [&] {
        launch_linear_f32_f8in_hrid_hout(att8.as<u8>(), p1.as<u8>(), ah.as<__half>(), n, C, C,
                                         bh.as<__half>(), s1.as<float>(), 4, 0, false);
    });
    time_it("launch_reframe_f32_hin", [&] {
        launch_reframe_f32_hin(ah.as<__half>(), skip0.as<float>(), W, Ht, W, Ht, 0, 0, C, 1, 0);
    });

    std::printf("\n  %-34s %7.3f ms   (block 00 reports 12.6, ~3%% inflated by its own timing)\n",
                "SUM with pageable upload", total);
    if (have_pinned)
        std::printf("  %-34s %7.3f ms   (%.3f ms saved, %.0f%% of block 0)\n",
                    "SUM with pinned upload", total - pageable + pinned,
                    pageable - pinned, 100.0 * (pageable - pinned) / total);
    if (have_pinned) hipHostFree(host_pinned);
    return 0;
}
