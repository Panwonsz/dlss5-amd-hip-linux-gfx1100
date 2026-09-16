// Fused QKV GEMM + per-head scaling + E4M3 quantization.
// Replaces k_linear_f32 (mode 3, ordered 0) followed by k_normalize_qkv for the
// window blocks: the f32 QKV tensor never touches global memory.
//
// Numerics are bit-identical to the split pair: the GEMM accumulation order is
// the same per-wave K loop, the per-row square sum keeps its strict channel
// order (one thread per row-head), and the E4M3 quantization sees the same
// f32 values (mode 3 applies no rounding between the two passes).
//
// Grid: (m/16, 3C/(32*NH)). A block covers NH heads (32*NH columns) with
// 2*NH waves; each wave owns one 16-column tile. One 32-channel head spans
// two adjacent waves.
#pragma once
#include "dlss5_linalg.hpp"

namespace dlss5 {

// Per-channel squares exactly as the split k_normalize_qkv computes them, without
// the volatile private-memory round trip. v_fma_mixlo_f16 with f32 operands
// multiplies at f32 precision and RNE-rounds the result to f16 — the bit-exact
// H(v*v) the split kernel compiles to (f16(v^2), not f16(f16(v)^2)). The opaque
// asm also blocks FMA contraction of the following accumulate.
__device__ inline float sq_half_f32(float v) {
    float t;
    asm("v_fma_mixlo_f16 %0, %1, %1, 0" : "=v"(t) : "v"(v));
    return __half2float(__ushort_as_half((unsigned short)(unsigned)__builtin_bit_cast(unsigned, t)));
}
__device__ inline float sq_f32(float v) {
    float t;
    asm("v_mul_f32_e32 %0, %1, %1" : "=v"(t) : "v"(v));
    return t;
}

template <int C, int NH, bool F8IN = false, bool HIN = false>
__global__ void k_qkv_norm_f32(const float* input, const u8* input_f8, const u8* weights,
                               u8* output, uint m, uint k, const float* scales, float qgain,
                               bool half_squares, const __half* input_h = nullptr) {
    using namespace linalg;
    constexpr uint HEADS = C / 32u;
    constexpr uint COLS = 32u * NH;
    const uint first = blockIdx.x * 16u;
    const uint head0 = blockIdx.y * NH;
    const uint wave = threadIdx.x / 32u;
    // Single LDS buffer: the A-tile staging (512B) aliases the start of the
    // stage buffer. The tile is only read inside the K loop; stage is only
    // written after it (with a barrier between), so the aliasing is safe.
    // Cuts LDS 512+16*COLS*4 -> 16*COLS*4 (e.g. C32/NH1 2560B->2048B).
    __shared__ float stage[16 * COLS];
    u8* tile = reinterpret_cast<u8*>(stage);
    C32 acc = C32::Splat(0.f);
    const uint col = blockIdx.y * COLS + wave * 16u;
    // blockDim = 2*NH*32, 8 threads per input row (float4). C32 uses NH=3 so one
    // CTA computes Q+K+V and reads the input tile once; then 192 threads exist but
    // only 16*8 load, one row each. NH=1 still needs two rows per thread.
    constexpr uint LOADERS = 16u * 8u;
    constexpr uint THREADS = 2u * NH * 32u;
    constexpr uint ROWS_PER_THREAD = (LOADERS + THREADS - 1u) / THREADS;
    for (uint g = 0; g < k / 32; ++g) {
        if (threadIdx.x < LOADERS) {
            const uint c4 = (threadIdx.x % 8u) * 4u;
            const uint base = threadIdx.x / 8u;
            for (uint rr = 0; rr < ROWS_PER_THREAD; ++rr) {
                const uint rlocal = base * ROWS_PER_THREAD + rr;
                const uint row = first + rlocal;
                if (F8IN) {
                    *reinterpret_cast<u32*>(tile + rlocal * 32u + c4) =
                        row < m
                            ? *reinterpret_cast<const u32*>(input_f8 + row * k + g * 32u + c4)
                            : 0u;
                } else if (HIN) {
                    const __half2* hp = reinterpret_cast<const __half2*>(
                        input_h + row * k + g * 32u + c4);
                    __half2 h0 = row < m ? hp[0] : __half2(0.f, 0.f);
                    __half2 h1 = row < m ? hp[1] : __half2(0.f, 0.f);
                    float x = __half2float(__low2half(h0));
                    float y = __half2float(__high2half(h0));
                    float z = __half2float(__low2half(h1));
                    float w = __half2float(__high2half(h1));
                    *reinterpret_cast<u32*>(tile + rlocal * 32u + c4) =
                        u32(e4m3_byte(x)) | u32(e4m3_byte(y)) << 8 |
                        u32(e4m3_byte(z)) << 16 | u32(e4m3_byte(w)) << 24;
                } else {
                    const float4 v = row < m
                        ? *reinterpret_cast<const float4*>(input + row * k + g * 32u + c4)
                        : make_float4(0.f, 0.f, 0.f, 0.f);
                    *reinterpret_cast<u32*>(tile + rlocal * 32u + c4) =
                        u32(e4m3_byte(v.x)) | u32(e4m3_byte(v.y)) << 8 |
                        u32(e4m3_byte(v.z)) << 16 | u32(e4m3_byte(v.w)) << 24;
                }
            }
        }
        __syncthreads();
        A8 a = A8::Load(tile, 0, 32);
        B8 b = B8::LoadRow(weights, ((col / 16u) * (k / 32u) + g) * 512u, 16);
        acc.MultiplyAccumulate(a, b);
        __syncthreads();
    }
    acc.StoreF32(stage, wave * 16u * 4u, COLS * 4u);
    __syncthreads();
    // 16 rows x NH heads; one thread per row-head keeps the strict channel sum order.
    const uint item = threadIdx.x;
    constexpr uint ROW_THREADS = 16 * NH;
    if (item < ROW_THREADS) {
        const uint row = item / NH;
        const uint headg = head0 + (item % NH);
        const float* s0 = &stage[row * COLS + (item % NH) * 32u];
        const float* s1 = s0 + 16u;
        float inv = 1.f;
        if (headg < 2u * HEADS) {
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
        }
        u32 w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint ch = 0; ch < 16u; ++ch)
            w[ch >> 2] |= u32(e4m3_byte(s0[ch] * inv)) << ((ch & 3u) * 8u);
        for (uint ch = 0; ch < 16u; ++ch)
            w[4u + (ch >> 2)] |= u32(e4m3_byte(s1[ch] * inv)) << ((ch & 3u) * 8u);
        u8* out = output + (size_t)(first + row) * (3u * C) + headg * 32u;
        uint4* o4 = reinterpret_cast<uint4*>(out);
        o4[0] = make_uint4(w[0], w[1], w[2], w[3]);
        o4[1] = make_uint4(w[4], w[5], w[6], w[7]);
    }
}

} // namespace dlss5
