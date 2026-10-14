// EXPERIMENT, NOT BIT-EXACT (DLSS5_F16_ACT=1, 2026-10-14): k_c32_fused with its activations kept in f16.
//
// k_c32_fused reproduces the network's FP8 (E4M3) activations exactly: every intermediate is rounded
// with e4m3_byte and decoded again for the next MMA. gfx1100 has no FP8 hardware, so those ~24k
// encodes per 64-token window are integer bit arithmetic, and they are what limits this kernel (the
// largest single bucket of the model, ~15 ms per frame). Here every point that rounds to E4M3 rounds to
// f16 instead, and the LDS tiles are f16, loaded with plain load_matrix_sync. The weights are untouched
// (the same f16 copies of the E4M3 weights), as are the f32 accumulations, the RMS, the softmax and the
// projection epilogue.
//
// So the output DIFFERS from the bit-exact model: activations carry more precision than the network
// was built with. Whether that is visible is what the experiment measures (tools/compare_f16act.py).
//
// Structure as k_c32_fused (one CTA per 8x8 window, 4 waves x 16 rows), with these LDS changes:
//   packed16  192 rows x PH halves   input -> Q -> attention out (0-63), K (64-127), V / FFN out (128-191)
//   scratch   raw f32 / scores f16 / probabilities f16, aliased as before
//   hidden16  64 x HS16 halves, overlaid on the whole block during the FFN (as DLSS5_C32_H16 did); the
//             FFN residual is therefore re-read from global
// Only the split-row epilogues (DLSS5_C32_SPLITROW=1, the default) are implemented. No debug outputs.
#pragma once
#include "dlss5_linalg.hpp"

namespace dlss5 {

template <bool W16, bool CHAIN, int OUTM, bool RASTER, bool RIN>
__global__ void __launch_bounds__(128) k_c32_fused_f16act(const __half* in_h, const u8* ffn_w,
                                                          const float* ffn_s, const u8* qkv_w,
                                                          const float* qscale, float qgain,
                                                          const float* bias, const u8* proj_w,
                                                          const float* proj_s, __half* out_ah,
                                                          __half* out_raster, uint w, uint h, uint sw,
                                                          uint px, uint py, float* out_f32,
                                                          uint bias_mode) {
    using namespace linalg;
    using BT = typename WeightB<W16>::type;
    using AF = typename HiddenA<true>::type; // f16-storage A fragment (see HiddenA)
    using BF = BW16;                          // f16-storage B fragment
    constexpr uint BSZ = W16 ? 2u : 1u;
    constexpr uint TILE = 512u * BSZ;
    constexpr uint TSTR = 16u * BSZ;
    constexpr uint PH = 40u;                  // packed16 row stride (halves): 32 + 8, 80 B
    constexpr uint HS16 = 136u;               // hidden16 row stride (halves)
    constexpr uint RS = 36u;                  // raw f32 row stride (floats)
    constexpr uint P16S = 72u;                // probability row stride (halves): 64 + 8, 144 B
    constexpr uint PACKED_BYTES = 192u * PH * 2u;                 // 15360
    constexpr uint SCRATCH_BYTES = 64u * P16S * 2u;               // 9216 (raw 9216, scores 8704)
    constexpr uint BASE_BYTES = PACKED_BYTES + SCRATCH_BYTES;     // 24576
    constexpr uint H16_BYTES = 64u * HS16 * 2u;                   // 17408
    constexpr uint LDS_TOTAL = H16_BYTES > BASE_BYTES ? H16_BYTES : BASE_BYTES;
    static_assert(64u * RS * 4u <= SCRATCH_BYTES && 64u * 68u * 2u <= SCRATCH_BYTES, "scratch");
    static_assert(DLSS5_C32_SPLITROW, "k_c32_fused_f16act implements the split-row epilogues only");
    __shared__ __align__(16) u8 lds[LDS_TOTAL];
    __half* packed16 = reinterpret_cast<__half*>(lds);
    __half* ffn16 = packed16 + 128u * PH;
    u8* scratch = lds + PACKED_BYTES;
    float* raw = reinterpret_cast<float*>(scratch);
    __half* scores = reinterpret_cast<__half*>(scratch);
    __half* p16 = reinterpret_cast<__half*>(scratch); // aliases scores after the row sums
    __half* hidden16 = reinterpret_cast<__half*>(lds);

    const uint window = blockIdx.x;
    const uint tid = threadIdx.x;
    const uint wave = tid / 32u;
    const uint lane = tid & 31u;
    const uint first = wave * 16u;

    // ---- prologue: 64 tokens x 32 f16, copied as they are ----------------------------------------
#pragma unroll
    for (uint it = 0; it < 4u; ++it) {
        const uint i4 = tid + it * 128u;
        const uint tok = i4 / 8u;
        const uint c4 = (i4 % 8u) * 4u;
        uint2 hv;
        if constexpr (RIN) {
            const uint x = (window % (sw / 8u)) * 8u + (tok & 7u);
            const uint y = (window / (sw / 8u)) * 8u + (tok >> 3u);
            hv = (x >= px && y >= py && x < w + px && y < h + py)
                     ? *reinterpret_cast<const uint2*>(in_h + (size_t(y - py) * w + (x - px)) * 32u + c4)
                     : make_uint2(0u, 0u);
        } else {
            hv = *reinterpret_cast<const uint2*>(in_h + (size_t(window) * 64u + tok) * 32u + c4);
        }
        *reinterpret_cast<uint2*>(packed16 + tok * PH + c4) = hv;
    }
    c32f_block_sync();

    // ---- FFN expand ----------------------------------------------------------------------------
    {
        const AF a = AF::Load(packed16, first * PH * 2u, PH * 2u);
        c32f_block_sync(); // hidden16 overlays packed16, which other waves may not have loaded yet
        for (uint ct = 0; ct < 8u; ++ct) {
            C32 ex = C32::Splat(0.f);
            const BT b = BT::LoadRow(ffn_w, ct * TILE, TSTR);
            ex.MultiplyAccumulate(a, b);
            for (uint i = 0; i < ex.Length(); ++i) {
                const uint2 c = ex.GetCoordinate(i);
                hidden16[(first + c.x) * HS16 + ct * 16u + c.y] = __float2half_rn(ActivatePolyC32(ex.Get(i)));
            }
        }
    }
    c32f_wave_sync();

    // ---- FFN contract: residual (the f16 input, from global) preset, 4 K-steps -------------------
    C32 acc0 = C32::Splat(0.f), acc1 = C32::Splat(0.f);
    for (uint i = 0; i < acc0.Length(); ++i) {
        const uint2 c = acc0.GetCoordinate(i);
        const uint row = first + c.x;
        const float iv0 = __half2float(c32f_input<RIN>(in_h, window, row, c.y, w, h, sw, px, py));
        const float iv1 = __half2float(c32f_input<RIN>(in_h, window, row, 16u + c.y, w, h, sw, px, py));
        acc0.Set(i, iv0 * ffn_s[c.y]);
        acc1.Set(i, iv1 * ffn_s[16u + c.y]);
    }
    for (uint g = 0; g < 4u; ++g) {
        const AF a = AF::Load(hidden16, (first * HS16 + g * 32u) * 2u, HS16 * 2u);
        const BT b0 = BT::LoadRow(ffn_w, 4096u * BSZ + (0u * 4u + g) * TILE, TSTR);
        const BT b1 = BT::LoadRow(ffn_w, 4096u * BSZ + (1u * 4u + g) * TILE, TSTR);
        acc0.MultiplyAccumulate(a, b0);
        acc1.MultiplyAccumulate(a, b1);
    }
    c32f_block_sync(); // every wave is done with hidden16 before packed16 is written again

    __half saved0[8], saved1[8];
    for (uint i = 0; i < acc0.Length(); ++i) {
        const uint2 c = acc0.GetCoordinate(i);
        const uint row = first + c.x;
        saved0[i] = __float2half_rn(acc0.Get(i));
        saved1[i] = __float2half_rn(acc1.Get(i));
        ffn16[row * PH + c.y] = saved0[i];
        ffn16[row * PH + 16u + c.y] = saved1[i];
    }
    c32f_wave_sync();

    // ---- QKV GEMM + Q/K RMS; V raw --------------------------------------------------------------
    for (uint part = 0; part < 3u; ++part) {
        const AF a = AF::Load(ffn16, first * PH * 2u, PH * 2u);
        const BT b0 = BT::LoadRow(qkv_w, (part * 2u + 0u) * TILE, TSTR);
        const BT b1 = BT::LoadRow(qkv_w, (part * 2u + 1u) * TILE, TSTR);
        C32 q0 = C32::Splat(0.f), q1 = C32::Splat(0.f);
        q0.MultiplyAccumulate(a, b0);
        q1.MultiplyAccumulate(a, b1);
        if (part < 2u) {
            for (uint i = 0; i < q0.Length(); ++i) {
                const uint2 c = q0.GetCoordinate(i);
                raw[(first + c.x) * RS + c.y] = q0.Get(i);
                raw[(first + c.x) * RS + 16u + c.y] = q1.Get(i);
            }
            c32f_wave_sync();
            {
                const uint row = first + (lane & 15u);
                const float* s0 = raw + row * RS;
                const float4* f0 = reinterpret_cast<const float4*>(s0);
                const float4* f1 = reinterpret_cast<const float4*>(s0 + 16u);
                float sum = 0.f;
#pragma unroll
                for (uint q = 0; q < 4u; ++q) {
                    const float4 v = f0[q];
                    sum += sq_half_f32(v.x);
                    sum += sq_half_f32(v.y);
                    sum += sq_half_f32(v.z);
                    sum += sq_half_f32(v.w);
                }
#pragma unroll
                for (uint q = 0; q < 4u; ++q) {
                    const float4 v = f1[q];
                    sum += sq_half_f32(v.x);
                    sum += sq_half_f32(v.y);
                    sum += sq_half_f32(v.z);
                    sum += sq_half_f32(v.w);
                }
                float inv = rsqrtf(fmaxf(sum, 6.198883056640625e-5f));
                if (part == 0u) inv *= qscale[0] * qgain;
                __half* dst = packed16 + (part * 64u + row) * PH;
                const uint hb = (lane >> 4u) * 16u;
                for (uint ch = 0; ch < 16u; ++ch) dst[hb + ch] = __float2half_rn(s0[hb + ch] * inv);
            }
            c32f_wave_sync();
        } else {
            for (uint i = 0; i < q0.Length(); ++i) {
                const uint2 c = q0.GetCoordinate(i);
                packed16[(128u + first + c.x) * PH + c.y] = __float2half_rn(q0.Get(i));
                packed16[(128u + first + c.x) * PH + 16u + c.y] = __float2half_rn(q1.Get(i));
            }
        }
    }
    c32f_block_sync();

    // ---- window attention -------------------------------------------------------------------------
    {
        const AF q = AF::Load(packed16, first * PH * 2u, PH * 2u);
        for (uint k0 = 0; k0 < 64u; k0 += 16u) {
            const BF k = BF::LoadCol(packed16, (64u + k0) * PH * 2u, PH * 2u);
            const C32 s = Multiply(q, k);
            float bv[8];
            if (bias_mode == 1u) {
                const float4* bp =
                    reinterpret_cast<const float4*>(bias) + ((wave * 4u + k0 / 16u) * 32u + lane) * 2u;
                const float4 x = bp[0], y = bp[1];
                bv[0] = x.x; bv[1] = x.y; bv[2] = x.z; bv[3] = x.w;
                bv[4] = y.x; bv[5] = y.y; bv[6] = y.z; bv[7] = y.w;
            } else if (bias_mode == 2u) {
                const uint4 x = reinterpret_cast<const uint4*>(bias)[(wave * 4u + k0 / 16u) * 32u + lane];
                const __half2* hh = reinterpret_cast<const __half2*>(&x);
#pragma unroll
                for (uint j = 0; j < 4u; ++j) {
                    bv[2u * j] = __half2float(__low2half(hh[j]));
                    bv[2u * j + 1u] = __half2float(__high2half(hh[j]));
                }
            }
            for (uint i = 0; i < s.Length(); ++i) {
                const uint2 c = s.GetCoordinate(i);
                const uint qi = first + c.x, kj = k0 + c.y;
                const float bb = bias_mode ? bv[i] : bias[qi * 64u + kj];
                scores[qi * 68u + kj] = __float2half_rn(fast_exp_raw(s.Get(i) + bb));
            }
        }
    }
    c32f_wave_sync();

    // Softmax: both halves of the wave sum the row (same order), each normalizes 32 of its 64 values.
    uint4 pr[4];
    {
        const uint qi = first + (lane & 15u);
        float sum = 0.f;
        for (int j = 0; j < 16; j++) {
            const uint2 v = *reinterpret_cast<const uint2*>(&scores[qi * 68u + j * 4]);
            const __half2* hh = reinterpret_cast<const __half2*>(&v);
            sum += __half2float(__low2half(hh[0]));
            sum += __half2float(__high2half(hh[0]));
            sum += __half2float(__low2half(hh[1]));
            sum += __half2float(__high2half(hh[1]));
        }
        const float inv = 1.f / fmaxf(sum, 1e-20f);
        const uint base = (lane >> 4u) * 32u;
        for (int q = 0; q < 4; q++) {
            __half* b = reinterpret_cast<__half*>(&pr[q]);
            for (int t = 0; t < 8; t++)
                b[t] = __float2half_rn(__half2float(scores[qi * 68u + base + q * 8 + t]) * inv);
        }
    }
    c32f_block_sync(); // p16 rows overlap other waves' not-yet-summed score rows
    {
        uint4* row = reinterpret_cast<uint4*>(p16 + (first + (lane & 15u)) * P16S + (lane >> 4u) * 32u);
        for (int q = 0; q < 4; q++) row[q] = pr[q];
    }
    c32f_wave_sync();

    {
        C32 av0 = C32::Splat(0.f), av1 = C32::Splat(0.f);
        for (uint k0 = 0; k0 < 64u; k0 += 32u) {
            const AF p = AF::Load(p16, (first * P16S + k0) * 2u, P16S * 2u);
            const BF v0 = BF::LoadRow(packed16, (128u + k0) * PH * 2u, PH * 2u);
            const BF v1 = BF::LoadRow(packed16, ((128u + k0) * PH + 16u) * 2u, PH * 2u);
            av0.MultiplyAccumulate(p, v0);
            av1.MultiplyAccumulate(p, v1);
        }
        for (uint i = 0; i < av0.Length(); ++i) {
            const uint2 c = av0.GetCoordinate(i);
            packed16[(first + c.x) * PH + c.y] = __float2half_rn(av0.Get(i));
            packed16[(first + c.x) * PH + 16u + c.y] = __float2half_rn(av1.Get(i));
        }
    }
    c32f_wave_sync();

    // ---- projection: as k_c32_fused ---------------------------------------------------------------
    {
        const AF a = AF::Load(packed16, first * PH * 2u, PH * 2u);
        const BT b0 = BT::LoadRow(proj_w, 0u, TSTR);
        const BT b1 = BT::LoadRow(proj_w, TILE, TSTR);
        C32 p0 = C32::Splat(0.f), p1 = C32::Splat(0.f);
        p0.MultiplyAccumulate(a, b0);
        p1.MultiplyAccumulate(a, b1);
        const uint wx0 = (window % (sw / 8u)) * 8u, wy0 = (window / (sw / 8u)) * 8u;
        for (uint i = 0; i < p0.Length(); ++i) {
            const uint2 c = p0.GetCoordinate(i);
            const uint row = first + c.x;
            float v0 = __builtin_fmaf(__half2float(saved0[i]), proj_s[c.y], p0.Get(i));
            float v1 = __builtin_fmaf(__half2float(saved1[i]), proj_s[16u + c.y], p1.Get(i));
#if defined(__HIP_DEVICE_COMPILE__)
            asm volatile("" : "+v"(v0), "+v"(v1));
#endif
            v0 = H(v0);
            v1 = H(v1);
            const __half h0 = __float2half_rn(v0), h1 = __float2half_rn(v1);
            if constexpr (OUTM == 1) {
                out_ah[(size_t(window) * 64u + row) * 32u + c.y] = h0;
                out_ah[(size_t(window) * 64u + row) * 32u + 16u + c.y] = h1;
            }
            if constexpr (RASTER || OUTM == 2) {
                const uint wx = wx0 + (row % 8u), wy = wy0 + (row / 8u);
                if (wx >= px && wy >= py && wx < w + px && wy < h + py) {
                    const size_t o = (size_t(wy - py) * w + (wx - px)) * 32u;
                    if constexpr (RASTER) {
                        out_raster[o + c.y] = h0;
                        out_raster[o + 16u + c.y] = h1;
                    }
                    if constexpr (OUTM == 2) {
                        out_f32[o + c.y] = __half2float(h0);
                        out_f32[o + 16u + c.y] = __half2float(h1);
                    }
                }
            }
        }
    }
}

} // namespace dlss5
