// C32 window block in one kernel: FFN -> QKV + per-head RMS -> window attention -> projection.
//
// Port of guentra's k_c32_fused (upstream 6a9040e / c32_fused.hip) to this fork's gfx11 rocWMMA
// linalg. One CTA = one 8x8 window (64 tokens), four waves, each wave owning 16 token rows. Every
// intermediate stays in LDS; only the f16 window output (and the f16 raster sibling for the next
// block) reach global memory.
//
// Replaces, for c == 32 with an f16 window input, the four launches
//   k_ffn_f32<32, ..., HIN, HOUT>          (bh = H(FFN))
//   k_qkv_norm_f32<32, 3, ..., HIN>        (q8 = E4M3 Q/K/V)
//   k_window_attention<32>                 (att8, direct_f8)
//   k_linear_f32 mode 4, HRID, HOUT[, HRAST] (ah [+ out16 raster])
// and must be BIT-IDENTICAL to them. Every numeric step below mirrors the split kernel it replaces:
//  - same weight buffers and tile offsets (fw/qw/p1 exactly as uploaded, E4M3 or W16 f16 tiles);
//  - same MMA order per 16x16 output tile (Splat, then MultiplyAccumulate in K order);
//  - FFN residual preset into the accumulator before the contract MMAs, as k_ffn_f32 does;
//  - projection residual added after the MMA (mode 4), then H(), as k_linear_f32 does;
//  - the same E4M3 quantizer (e4m3_byte) at every point the split path writes E4M3.
// Differences from upstream: this is the gfx11 accumulator layout (row = 2i + (lane >= 16)), so every
// element address comes from GetCoordinate(), never the gfx12 hard-coded row map; weights stay in
// the fork's tiled layout (no row-major repack); the QKV raw stage is one 36-float row per token
// reused across the three parts (upstream's part*32 offset ran 236 B past its LDS array).
#pragma once
#include "dlss5_linalg.hpp"

namespace dlss5 {

// Wave-local LDS ordering: a wave reading rows that other lanes of the SAME wave just wrote. DS
// operations of one wave execute in order; this keeps the compiler from moving them across.
__device__ inline void c32f_wave_sync() {
#if defined(__HIP_DEVICE_COMPILE__)
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#endif
}

template <bool W16, bool CHAIN, bool WRITE_AH, bool RASTER>
__global__ void __launch_bounds__(128) k_c32_fused(const __half* in_h, const u8* ffn_w,
                                                   const float* ffn_s, const u8* qkv_w,
                                                   const float* qscale, float qgain,
                                                   const float* bias, const u8* proj_w,
                                                   const float* proj_s, __half* out_ah,
                                                   __half* out_raster, uint w, uint h, uint sw,
                                                   uint px, uint py, __half* dbg_bh, u8* dbg_q8,
                                                   u8* dbg_att8) {
    using namespace linalg;
    using BT = typename WeightB<W16>::type;
    constexpr uint BSZ = W16 ? 2u : 1u;   // bytes per weight element
    constexpr uint TILE = 512u * BSZ;     // one 16x32 weight tile
    constexpr uint TSTR = 16u * BSZ;      // its row stride
    constexpr uint PS = 36u;              // packed E4M3 row stride (bytes)
    constexpr uint HS = 132u;             // hidden E4M3 row stride (bytes)
    constexpr uint RS = 36u;              // raw f32 row stride (floats)
    constexpr uint PACKED_BYTES = 3u * 64u * PS;   // 6912
    constexpr uint SCRATCH_BYTES = 64u * RS * 4u;  // 9216 (hidden 8448, raw 9216, scores 8704)

    __shared__ __align__(16) u8 lds[PACKED_BYTES + SCRATCH_BYTES];
    // packed rows 0-63: input E4M3, then Q, then the attention output; 64-127: K; 128-191: V.
    u8* packed = lds;
    // FFN output E4M3 (the QKV GEMM's A), in the V rows until V overwrites them.
    u8* ffn8 = packed + 128u * PS;
    // f16 FFN residual for non-chain blocks, in the K/V rows during the FFN (row stride 34 halves).
    __half* in16 = reinterpret_cast<__half*>(packed + 64u * PS);
    u8* scratch = lds + PACKED_BYTES;
    u8* hidden = scratch;
    float* raw = reinterpret_cast<float*>(scratch);
    __half* scores = reinterpret_cast<__half*>(scratch);
    u8* p8 = scratch;  // aliases scores after the row sums (barrier 4)

    const uint window = blockIdx.x;
    const uint tid = threadIdx.x;
    const uint wave = tid / 32u;
    const uint lane = tid & 31u;
    const uint first = wave * 16u;

    // ---- prologue: 64 tokens x 32 f16 -> E4M3 bytes (+ f16 residual copy) ----------------------
#pragma unroll
    for (uint it = 0; it < 4u; ++it) {
        const uint i4 = tid + it * 128u;
        const uint tok = i4 / 8u;
        const uint c4 = (i4 % 8u) * 4u;
        const uint2 hv =
            *reinterpret_cast<const uint2*>(in_h + (size_t(window) * 64u + tok) * 32u + c4);
        const __half2* hp = reinterpret_cast<const __half2*>(&hv);
        if constexpr (!CHAIN) {
            *reinterpret_cast<__half2*>(in16 + tok * 34u + c4) = hp[0];
            *reinterpret_cast<__half2*>(in16 + tok * 34u + c4 + 2u) = hp[1];
        }
        *reinterpret_cast<u32*>(packed + tok * PS + c4) =
            u32(e4m3_byte(__half2float(__low2half(hp[0])))) |
            u32(e4m3_byte(__half2float(__high2half(hp[0])))) << 8 |
            u32(e4m3_byte(__half2float(__low2half(hp[1])))) << 16 |
            u32(e4m3_byte(__half2float(__high2half(hp[1])))) << 24;
    }
    // Barrier 1: each wave's rows were staged by all four waves.
    __syncthreads();

    // ---- FFN expand: 16 rows x 128 hidden per wave, ActivatePolyC32, E4M3 -----------------------
    {
        const A8 a = A8::Load(packed + first * PS, 0, PS);
        for (uint ct = 0; ct < 8u; ++ct) {
            C32 ex = C32::Splat(0.f);
            const BT b = BT::LoadRow(ffn_w, ct * TILE, TSTR);
            ex.MultiplyAccumulate(a, b);
            for (uint i = 0; i < ex.Length(); ++i) {
                const uint2 c = ex.GetCoordinate(i);
                hidden[(first + c.x) * HS + ct * 16u + c.y] = e4m3_byte(ActivatePolyC32(ex.Get(i)));
            }
        }
    }
    c32f_wave_sync();

    // ---- FFN contract: residual preset, then 4 K-steps (k_ffn_f32 order) ------------------------
    C32 acc0 = C32::Splat(0.f), acc1 = C32::Splat(0.f);
    for (uint i = 0; i < acc0.Length(); ++i) {
        const uint2 c = acc0.GetCoordinate(i);
        const uint row = first + c.x;
        float iv0, iv1;
        if constexpr (CHAIN) {
            // from_e4m3(e4m3_byte(input)) -- the byte is already in packed.
            iv0 = from_e4m3(packed[row * PS + c.y]);
            iv1 = from_e4m3(packed[row * PS + 16u + c.y]);
        } else {
            iv0 = __half2float(in16[row * 34u + c.y]);
            iv1 = __half2float(in16[row * 34u + 16u + c.y]);
        }
        acc0.Set(i, iv0 * ffn_s[c.y]);
        acc1.Set(i, iv1 * ffn_s[16u + c.y]);
    }
    for (uint g = 0; g < 4u; ++g) {
        const A8 a = A8::Load(hidden + first * HS, g * 32u, HS);
        const BT b0 = BT::LoadRow(ffn_w, 4096u * BSZ + (0u * 4u + g) * TILE, TSTR);
        const BT b1 = BT::LoadRow(ffn_w, 4096u * BSZ + (1u * 4u + g) * TILE, TSTR);
        acc0.MultiplyAccumulate(a, b0);
        acc1.MultiplyAccumulate(a, b1);
    }
    // Barrier 2: ffn8 (V rows) overlaps in16, which every wave read above.
    __syncthreads();

    // bh = H(FFN) in f16 (kept for the projection residual); ffn8 = its E4M3 (QKV input).
    __half saved0[8], saved1[8];
    for (uint i = 0; i < acc0.Length(); ++i) {
        const uint2 c = acc0.GetCoordinate(i);
        const uint row = first + c.x;
        saved0[i] = __float2half_rn(acc0.Get(i));
        saved1[i] = __float2half_rn(acc1.Get(i));
        ffn8[row * PS + c.y] = e4m3_byte(__half2float(saved0[i]));
        ffn8[row * PS + 16u + c.y] = e4m3_byte(__half2float(saved1[i]));
        if (dbg_bh) {  // DLSS5_C32_CHECK: the split path's bh
            dbg_bh[(size_t(window) * 64u + row) * 32u + c.y] = saved0[i];
            dbg_bh[(size_t(window) * 64u + row) * 32u + 16u + c.y] = saved1[i];
        }
    }
    c32f_wave_sync();

    // ---- QKV GEMM + Q/K RMS (k_qkv_norm_f32<32,3> numerics); V raw E4M3 --------------------------
    for (uint part = 0; part < 3u; ++part) {
        const A8 a = A8::Load(ffn8 + first * PS, 0, PS);
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
            if (lane < 16u) {
                const uint row = first + lane;
                const float* s0 = raw + row * RS;
                const float* s1 = s0 + 16u;
                const float4* f0 = reinterpret_cast<const float4*>(s0);
                const float4* f1 = reinterpret_cast<const float4*>(s1);
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
                if (part == 0u)
                    inv *= qscale[0] * qgain;
                u8* dst = packed + (part * 64u + row) * PS;
                for (uint ch = 0; ch < 16u; ++ch) dst[ch] = e4m3_byte(s0[ch] * inv);
                for (uint ch = 0; ch < 16u; ++ch) dst[16u + ch] = e4m3_byte(s1[ch] * inv);
                if (dbg_q8)  // DLSS5_C32_CHECK: the split path's q8 (E4M3 layout [token][96])
                    for (uint ch = 0; ch < 32u; ++ch)
                        dbg_q8[(size_t(window) * 64u + row) * 96u + part * 32u + ch] = dst[ch];
            }
            c32f_wave_sync();
        } else {
            // V: no normalization, E4M3 of the raw GEMM (the V-tail of k_qkv_norm_f32). The rows are
            // this wave's own ffn8 rows, already consumed into `a`.
            for (uint i = 0; i < q0.Length(); ++i) {
                const uint2 c = q0.GetCoordinate(i);
                packed[(128u + first + c.x) * PS + c.y] = e4m3_byte(q0.Get(i));
                packed[(128u + first + c.x) * PS + 16u + c.y] = e4m3_byte(q1.Get(i));
                if (dbg_q8) {
                    const size_t o = (size_t(window) * 64u + first + c.x) * 96u + 64u;
                    dbg_q8[o + c.y] = e4m3_byte(q0.Get(i));
                    dbg_q8[o + 16u + c.y] = e4m3_byte(q1.Get(i));
                }
            }
        }
    }
    // Barrier 3: QK reads every wave's K rows, PV every wave's V rows; scores overwrite raw.
    __syncthreads();

    // ---- window attention (k_window_attention<32>, direct_f8) ------------------------------------
    {
        const A8 q = A8::Load(packed + first * PS, 0, PS);
        for (uint k0 = 0; k0 < 64u; k0 += 16u) {
            const B8 k = B8::LoadCol(packed, (64u + k0) * PS, PS);
            const C32 s = Multiply(q, k);
            for (uint i = 0; i < s.Length(); ++i) {
                const uint2 c = s.GetCoordinate(i);
                const uint qi = first + c.x, kj = k0 + c.y;
                scores[qi * 68u + kj] = __float2half_rn(fast_exp_raw(s.Get(i) + bias[qi * 64u + kj]));
            }
        }
    }
    c32f_wave_sync();

    uint4 p8r[4];
    if (lane < 16u) {
        const uint qi = first + lane;
        float sum = 0.f;
        for (int j = 0; j < 16; j++) {
            const uint2 v = *reinterpret_cast<const uint2*>(&scores[qi * 68u + j * 4]);
            const __half2* hh = reinterpret_cast<const __half2*>(&v);
            sum += __half2float(__low2half(hh[0]));
            sum += __half2float(__high2half(hh[0]));
            sum += __half2float(__low2half(hh[1]));
            sum += __half2float(__high2half(hh[1]));
        }
        // IEEE 1/x: bit-identical to the rcp+Newton form on gfx1100 (test_norm_inverse.hip).
        const float inv = 1.f / fmaxf(sum, 1e-20f);
        for (int q4 = 0; q4 < 4; q4++) {
            uint4 v;
            u8* b = reinterpret_cast<u8*>(&v);
            for (int t = 0; t < 16; t++)
                b[t] = e4m3_byte(__half2float(scores[qi * 68u + q4 * 16 + t]) * inv);
            p8r[q4] = v;
        }
    }
    // Barrier 4: p8 rows (64 B) overlap other waves' not-yet-summed scores rows (136 B).
    __syncthreads();
    if (lane < 16u) {
        uint4* row = reinterpret_cast<uint4*>(p8 + (first + lane) * 64u);
        for (int q4 = 0; q4 < 4; q4++) row[q4] = p8r[q4];
    }
    c32f_wave_sync();

    {
        C32 av0 = C32::Splat(0.f), av1 = C32::Splat(0.f);
        for (uint k0 = 0; k0 < 64u; k0 += 32u) {
            const A8 p = A8::Load(p8, first * 64u + k0, 64);
            const B8 v0 = B8::LoadRow(packed + (128u + k0) * PS, 0, PS);
            const B8 v1 = B8::LoadRow(packed + (128u + k0) * PS, 16u, PS);
            av0.MultiplyAccumulate(p, v0);
            av1.MultiplyAccumulate(p, v1);
        }
        // Attention output (E4M3, direct_f8) into this wave's Q rows, which QK has consumed.
        for (uint i = 0; i < av0.Length(); ++i) {
            const uint2 c = av0.GetCoordinate(i);
            packed[(first + c.x) * PS + c.y] = e4m3_byte(av0.Get(i));
            packed[(first + c.x) * PS + 16u + c.y] = e4m3_byte(av1.Get(i));
            if (dbg_att8) {  // DLSS5_C32_CHECK: the split path's att8
                const size_t o = (size_t(window) * 64u + first + c.x) * 32u;
                dbg_att8[o + c.y] = e4m3_byte(av0.Get(i));
                dbg_att8[o + 16u + c.y] = e4m3_byte(av1.Get(i));
            }
        }
    }
    c32f_wave_sync();

    // ---- projection (k_linear_f32 mode 4, HRID, HOUT[, HRAST]) -----------------------------------
    {
        const A8 a = A8::Load(packed + first * PS, 0, PS);
        const BT b0 = BT::LoadRow(proj_w, 0u, TSTR);
        const BT b1 = BT::LoadRow(proj_w, TILE, TSTR);
        C32 p0 = C32::Splat(0.f), p1 = C32::Splat(0.f);
        p0.MultiplyAccumulate(a, b0);
        p1.MultiplyAccumulate(a, b1);
        const uint wx0 = (window % (sw / 8u)) * 8u, wy0 = (window / (sw / 8u)) * 8u;
        for (uint i = 0; i < p0.Length(); ++i) {
            const uint2 c = p0.GetCoordinate(i);
            const uint row = first + c.x;
            // k_linear_f32 mode 4 computes v = fma(residual, scale, acc) in f32 (v_fma_mix_f32), then
            // H(v): two roundings. Written plainly here, the compiler fuses fma + H into one
            // v_fma_mixlo_f16 -- a single rounding straight to f16 -- which differs by one f16 ulp
            // on ~0.2% of outputs (DLSS5_C32_CHECK: bh/q8/att8 ok, ah not). The empty asm pins the
            // f32 result so the two roundings stay separate.
            float v0 = __builtin_fmaf(__half2float(saved0[i]), proj_s[c.y], p0.Get(i));
            float v1 = __builtin_fmaf(__half2float(saved1[i]), proj_s[16u + c.y], p1.Get(i));
#if defined(__HIP_DEVICE_COMPILE__)
            asm volatile("" : "+v"(v0), "+v"(v1));
#endif
            v0 = H(v0);
            v1 = H(v1);
            const __half h0 = __float2half_rn(v0), h1 = __float2half_rn(v1);
            if constexpr (WRITE_AH) {
                out_ah[(size_t(window) * 64u + row) * 32u + c.y] = h0;
                out_ah[(size_t(window) * 64u + row) * 32u + 16u + c.y] = h1;
            }
            if constexpr (RASTER) {
                const uint wx = wx0 + (row % 8u), wy = wy0 + (row / 8u);
                if (wx >= px && wy >= py && wx < w + px && wy < h + py) {
                    const size_t o = (size_t(wy - py) * w + (wx - px)) * 32u;
                    out_raster[o + c.y] = h0;
                    out_raster[o + 16u + c.y] = h1;
                }
            }
        }
    }
}

} // namespace dlss5
