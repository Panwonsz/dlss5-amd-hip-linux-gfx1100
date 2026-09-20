// What block 70's 10.8 ms is actually made of.
//
// Block 70 is the only stage that got SLOWER across the 217 -> 98 ms campaign (12.8 -> 13.3 ms
// together with the head, +4%, while every other stage fell 14-53%). It is also the only large
// region of the frame with no per-kernel breakdown, so "the post70 merge is slow" has been a label
// rather than a finding. This splits it.
//
// Its interval is bounded by mark(69) and mark(70) in network.hip, which makes it seven kernels:
// launch_post_merge_f32_h, then blocks[70].run(), which is a leading reframe + the same five-kernel
// C32 window block 0 runs + a trailing reframe.
//
// Three things make it NOT a copy of bench-block0, and they are the reason this file exists:
//   - block 70 has shift=3, so sw=1928 sh=1160 and n=2,236,480 -- 1.1% more tokens than block 0's
//     2,211,840, on top of two kernels block 0 does not run.
//   - chain=false here (id 70 is not in the chain set), where block 0's C32 siblings have it on.
//   - its input arrives as f16 (s.out16), so the f32 buffer post_merge writes is passed to
//     Block::run and then, in the c==32 in_h path, never read. See the post_merge note below.
//
// The weights are synthetic: this measures time, not numbers. Every kernel here has correctness
// cover elsewhere (test_linear, test_ffn_f32, test_attention, hip-network70's mean).
//
// The self-check is the sum, against the PROFILER's 10.8 -- not against another bench in this
// directory. bench-attention agreed with bench-block0 to 0.3% and both were measuring the same
// wrong half of a producer/consumer pair; two instruments agreeing with each other is not evidence.
//
//   usage: ./bench-block70 [iters]
#include "dlss5_launch.hpp"
#include "dlss5_runtime.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace dlss5;

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 20;
    const uint W = 1920, Ht = 1152, C = 32;
    const uint px = 4, py = 4;                       // block 70: shift=3
    const uint sw = (W + px + 7) & ~7u;              // 1928
    const uint sh = (Ht + py + 7) & ~7u;             // 1160
    const uint n = sw * sh;                          // 2,236,480
    const size_t N = size_t(W) * Ht;                 // 2,211,840, the unpadded raster
    const size_t nsc = size_t(W + 8) * (Ht + 8) * 32;

    hip_check(hipSetDevice(0), "device");
    hipDeviceProp_t prop{};
    hip_check(hipGetDeviceProperties(&prop, 0), "props");
    std::printf("bench-block70 device=%s arch=%s weights=%s kernels[%s] tokens=%u (block 0 runs %u) iters=%d\n",
                prop.name, prop.gcnArchName, dlss5_w16() ? "f16" : "e4m3",
                dlss5_kernel_formats(), n, W * Ht, iters);

    // ---- weights, in whichever format the kernels will read them ------------------------------
    auto fill = [](std::vector<float>& v, unsigned salt) {
        for (size_t i = 0; i < v.size(); ++i)
            v[i] = float(int((i * 7919u + salt) % 17) - 8) * 0.03125f;
    };
    const size_t m = size_t(C) * C;
    std::vector<float> ffn(8 * m), qkv(3 * m), proj(m);
    fill(ffn, 1); fill(qkv, 2); fill(proj, 3);

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
    DeviceBuf post_sc = floats_of(64, 0.9f), head_rgb = floats_of(96, 0.05f);

    // ---- buffers -------------------------------------------------------------------------------
    // lo is read at half resolution (lp maxes at 552,959) but is allocated full-size; pre and the
    // merged f32 output are the network's two full-raster ping-pong buffers.
    DeviceBuf lo(N * 32 * 4), pre(N * 32 * 4), merged(N * 32 * 4), merged_h(N * 32 * 2);
    DeviceBuf a16(nsc * 2), bh(nsc * 2), q8(nsc * 3 * (DLSS5_QKV16 ? 2 : 1)), att8(nsc), ah(nsc * 2);
    DeviceBuf b70_out(N * 32 * 4), color(N * 16), rgb_out(N * 12);

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
        std::printf("  %-36s %7.3f ms\n", name, ms);
        return ms;
    };

    std::printf("\nkernels, in the order block 70 runs them:\n");
    double merge_h = time_it("launch_post_merge_f32_h", [&] {
        launch_post_merge_f32_h(lo.as<float>(), pre.as<float>(), post_sc.as<float>(),
                                merged.as<float>(), merged_h.as<__half>(), W, Ht, true);
    });
    time_it("launch_reframe_hin_hout", [&] {
        launch_reframe_hin_hout(merged_h.as<__half>(), a16.as<__half>(), W, Ht, sw, sh, px, py, C);
    });
    time_it("launch_ffn_f32_hin_hout", [&] {
        launch_ffn_f32_hin_hout(32, a16.as<__half>(), fw.as<u8>(), fs.as<float>(),
                                bh.as<__half>(), n, false, true);   // chain=false at block 70
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
        launch_reframe_f32_hin(ah.as<__half>(), b70_out.as<float>(), W, Ht, sw, sh, px, py, C, 1, 0);
    });

    std::printf("\n  %-36s %7.3f ms   (block 70 reports 10.8, ~3%% inflated by its own timing)\n",
                "SUM", total);

    // ---- the merge's two stores, separately ----------------------------------------------------
    // k_post_merge_f32 stores either output only when given a pointer for it, so all three
    // combinations are measurable. That matters because block 70 takes the in_h path, where
    // Block::run hands window() the merged f32 buffer as `in` and the c==32 branch reads only
    // in_h -- the f32 store is dead there, and network.hip now passes nullptr for it.
    //
    // An earlier version of this section did NOT measure the f16-only case. It differenced the two
    // launchers and scaled the result by byte count, which priced the f16 store at 1863 GB/s on an
    // 800 GB/s card. A marginal-byte model that returns 2.3x the physical peak is broken, and the
    // 0.153 ms it predicted was three times too small. The row that matters is measured now.
    std::printf("\nthe merge's two stores (diagnostic, not part of the sum):\n");
    const double before = total;
    double merge_f32 = time_it("f32 store only", [&] {
        launch_post_merge_f32(lo.as<float>(), pre.as<float>(), post_sc.as<float>(),
                              merged.as<float>(), W, Ht, true);
    });
    double merge_f16 = time_it("f16 store only  <- what 70 now runs", [&] {
        launch_post_merge_f32_h(lo.as<float>(), pre.as<float>(), post_sc.as<float>(),
                                nullptr, merged_h.as<__half>(), W, Ht, true);
    });
    double merge_none = time_it("neither store (the read side alone)", [&] {
        launch_post_merge_f32_h(lo.as<float>(), pre.as<float>(), post_sc.as<float>(),
                                nullptr, nullptr, W, Ht, true);
    });
    total = before;
    std::printf("  %-36s %7.3f ms   (%.1f%% of block 70)\n",
                "dropping the dead f32 store saves", merge_h - merge_f16,
                100.0 * (merge_h - merge_f16) / total);
    std::printf("  %-36s %7.3f ms   <- the floor: reads + arithmetic, no stores at all\n",
                "if that looks small, the read side is", merge_none);

    // ---- what follows block 70, for context ----------------------------------------------------
    // The profile's separate "head + download" row, 2.5 ms. Not part of block 70's interval.
    std::printf("\nafter block 70 (the profile's separate 2.5 ms row):\n");
    std::vector<float> host(N * 3);
    time_it("launch_head_f32", [&] {
        launch_head_f32(b70_out.as<float>(), head_rgb.as<float>(), color.as<float>(),
                        rgb_out.as<float>(), uint(N));
    });
    time_it("hipMemcpyAsync D2H pageable", [&] {
        hip_check(hipMemcpyAsync(host.data(), rgb_out.p, N * 12, hipMemcpyDeviceToHost, stream),
                  "D2H");
    });
    return 0;
}
