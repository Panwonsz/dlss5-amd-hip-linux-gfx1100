// QKV GEMM + per-head RMS normalization + E4M3, one CTA per 16 tokens covering ALL 3C columns
// (DLSS5_QKV_WIDE, 2026-09-28).
//
// k_qkv_norm_pair runs (C/32 + C/64) CTAs per 16 tokens, each staging the same input tile and doing
// only k/32 MMAs per wave: at C64 that is 2 MMAs per wave, and the kernel runs at ~15 TFLOPS against
// the FFN's ~31. Here one CTA (4 waves) stages the 16 x C input tile ONCE for the whole K, then each
// wave owns 3C/64 column tiles, and the Q/K normalization runs on 16 x 2C/32 threads at once.
//
// Bit-identical to k_qkv_norm_f32 / k_qkv_norm_pair, by construction:
//  - the tile is staged with the same code (same E4M3 codes, same f16 when AH16);
//  - every 16-column accumulator starts at zero and receives the same MMAs (same a, same b) in the
//    same g order;
//  - the Q/K epilogue is the same code per (row, head): strict channel-order square sum, rsqrt, the
//    scale for Q heads, E4M3 codes, the same stores;
//  - V tiles are stored exactly as the V-only path stores them.
#pragma once
#include "dlss5_linalg.hpp"

namespace dlss5 {

template <int C, bool F8IN, bool HIN, bool AH16, bool W16, bool QOUT16 = DLSS5_QKV16>
__global__ void __launch_bounds__(128) k_qkv_norm_wide(const float* input, const u8* input_f8,
                                                       const u8* weights, u8* output, uint m, uint k,
                                                       const float* scales, float qgain,
                                                       bool half_squares, const __half* input_h) {
    using namespace linalg;
    constexpr uint HEADS = C / 32u;
    constexpr uint TILES = 3u * C / 16u;       // 16-column output tiles
    constexpr uint TPW = TILES / 4u;           // per wave
    constexpr uint QK_TILES = 4u * HEADS;      // Q and K: two tiles per head
    constexpr uint STAGE_COLS = 2u * C;        // Q and K columns, f32
    constexpr uint ASZ = AH16 ? 2u : 1u;
    static_assert(TILES % 4u == 0u, "tiles per wave");
    static_assert(16u * STAGE_COLS * 4u >= 16u * C * ASZ, "A tile must fit inside stage");
    __shared__ __align__(16) float stage[16 * STAGE_COLS];
    u8* tile = reinterpret_cast<u8*>(stage);  // 16 rows x C, aliased (read only before the stage writes)
    const uint first = blockIdx.x * 16u;
    const uint wave = threadIdx.x / 32u;

    // ---- stage the whole 16 x C input tile, formed exactly as k_qkv_norm_f32 forms each 16 x 32 ----
    for (uint i = threadIdx.x; i < 16u * (C / 4u); i += blockDim.x) {
        const uint rlocal = i / (C / 4u), c4 = (i % (C / 4u)) * 4u;
        const uint row = first + rlocal;
        u32 q;
        if (F8IN) {
            q = row < m ? *reinterpret_cast<const u32*>(input_f8 + row * k + c4) : 0u;
        } else if (HIN) {
            const __half2* hp = reinterpret_cast<const __half2*>(input_h + row * k + c4);
            __half2 h0 = row < m ? hp[0] : __half2(0.f, 0.f);
            __half2 h1 = row < m ? hp[1] : __half2(0.f, 0.f);
            float x = __half2float(__low2half(h0));
            float y = __half2float(__high2half(h0));
            float z = __half2float(__low2half(h1));
            float w = __half2float(__high2half(h1));
            q = u32(e4m3_byte(x)) | u32(e4m3_byte(y)) << 8 |
                u32(e4m3_byte(z)) << 16 | u32(e4m3_byte(w)) << 24;
        } else {
            const float4 v = row < m
                ? *reinterpret_cast<const float4*>(input + row * k + c4)
                : make_float4(0.f, 0.f, 0.f, 0.f);
            q = u32(e4m3_byte(v.x)) | u32(e4m3_byte(v.y)) << 8 |
                u32(e4m3_byte(v.z)) << 16 | u32(e4m3_byte(v.w)) << 24;
        }
        u8* dst = tile + (rlocal * C + c4) * ASZ;
        if constexpr (AH16)
            *reinterpret_cast<uint2*>(dst) = e4m3x4_to_half4(q);
        else
            *reinterpret_cast<u32*>(dst) = q;
    }
    __syncthreads();

    // ---- GEMM: wave w owns tiles w*TPW .. w*TPW+TPW-1 ----------------------------------------------
    using AT = typename HiddenA<AH16>::type;
    constexpr uint BSZ = W16 ? 2u : 1u;
    using BT = typename WeightB<W16>::type;
    C32 acc[TPW];
#pragma unroll
    for (uint t = 0; t < TPW; ++t) acc[t] = C32::Splat(0.f);
    for (uint g = 0; g < C / 32u; ++g) {
        AT a = AT::Load(tile, g * 32u * ASZ, C * ASZ);
#pragma unroll
        for (uint t = 0; t < TPW; ++t) {
            const uint col = (wave * TPW + t) * 16u;
            BT b = BT::LoadRow(weights, ((col / 16u) * (k / 32u) + g) * 512u * BSZ, 16 * BSZ);
            acc[t].MultiplyAccumulate(a, b);
        }
    }
    __syncthreads();  // everyone is done reading the tile before the stage (which aliases it) is written

    // ---- Q/K tiles -> stage; V tiles -> output directly (the V-only path's store) -----------------
#pragma unroll
    for (uint t = 0; t < TPW; ++t) {
        const uint tg = wave * TPW + t;
        if (tg < QK_TILES) {
            acc[t].StoreF32(stage, tg * 16u * 4u, STAGE_COLS * 4u);
        } else {
            const uint headg = tg / 2u;
            u8* out;
            if constexpr (QOUT16)
                out = output + ((size_t)first * (3u * C) + headg * 32u + (tg % 2u) * 16u) * 2u;
            else
                out = output + (size_t)first * (3u * C) + headg * 32u + (tg % 2u) * 16u;
            for (uint i = 0; i < acc[t].Length(); ++i) {
                uint2 rc = acc[t].GetCoordinate(i);
                if (first + rc.x < m) {
                    const u8 q = e4m3_byte(acc[t].Get(i));
                    if constexpr (QOUT16)
                        reinterpret_cast<uint16_t*>(out)[(size_t)rc.x * (3u * C) + rc.y] =
                            e4m3_to_half_u16(q);
                    else
                        out[(size_t)rc.x * (3u * C) + rc.y] = q;
                }
            }
        }
    }
    __syncthreads();

    // ---- normalization: one thread per (row, Q/K head), the k_qkv_norm_f32 epilogue --------------
    constexpr uint NORM_HEADS = 2u * HEADS;
    for (uint item = threadIdx.x; item < 16u * NORM_HEADS; item += blockDim.x) {
        const uint row = item / NORM_HEADS;
        const uint headg = item % NORM_HEADS;
        const float* s0 = &stage[row * STAGE_COLS + headg * 32u];
        const float* s1 = s0 + 16u;
        float inv = 1.f;
        float sum = 0.f;
        const float4* q0 = reinterpret_cast<const float4*>(s0);
        const float4* q1 = reinterpret_cast<const float4*>(s1);
        #pragma unroll
        for (uint q = 0; q < 4u; ++q) {
            float4 a = q0[q];
            sum += half_squares ? sq_half_f32(a.x) : sq_f32(a.x);
            sum += half_squares ? sq_half_f32(a.y) : sq_f32(a.y);
            sum += half_squares ? sq_half_f32(a.z) : sq_f32(a.z);
            sum += half_squares ? sq_half_f32(a.w) : sq_f32(a.w);
        }
        #pragma unroll
        for (uint q = 0; q < 4u; ++q) {
            float4 b = q1[q];
            sum += half_squares ? sq_half_f32(b.x) : sq_f32(b.x);
            sum += half_squares ? sq_half_f32(b.y) : sq_f32(b.y);
            sum += half_squares ? sq_half_f32(b.z) : sq_f32(b.z);
            sum += half_squares ? sq_half_f32(b.w) : sq_f32(b.w);
        }
        inv = rsqrtf(fmaxf(sum, 6.198883056640625e-5f));
        if (headg < HEADS)
            inv *= scales[headg] * qgain;
        u32 w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint ch = 0; ch < 16u; ++ch)
            w[ch >> 2] |= u32(e4m3_byte(s0[ch] * inv)) << ((ch & 3u) * 8u);
        for (uint ch = 0; ch < 16u; ++ch)
            w[4u + (ch >> 2)] |= u32(e4m3_byte(s1[ch] * inv)) << ((ch & 3u) * 8u);
        u8* out;
        if constexpr (QOUT16)
            out = output + ((size_t)(first + row) * (3u * C) + headg * 32u) * 2u;
        else
            out = output + (size_t)(first + row) * (3u * C) + headg * 32u;
        uint4* o4 = reinterpret_cast<uint4*>(out);
        if constexpr (QOUT16) {
            for (uint j = 0; j < 4u; ++j) {
                uint2 a = e4m3x4_to_half4(w[j * 2u]), b = e4m3x4_to_half4(w[j * 2u + 1u]);
                o4[j] = make_uint4(a.x, a.y, b.x, b.y);
            }
        } else {
            o4[0] = make_uint4(w[0], w[1], w[2], w[3]);
            o4[1] = make_uint4(w[4], w[5], w[6], w[7]);
        }
    }
}

} // namespace dlss5
