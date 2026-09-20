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

// AH16: the A-tile is staged as the f16 of the E4M3-rounded input rather than as E4M3 bytes, so the
// A fragments load with one rocwmma::load_matrix_sync instead of load_e4m3's per-lane unpack. The
// staging rounds to E4M3 either way and every E4M3 code is exact in f16, so the matrix unit sees the
// same numbers and the output hash is unchanged (tests/bench_qkv32.hip).
// W16: the weights arrive as f16 rather than E4M3, so the B fragments load with one
// load_matrix_sync too. AH16 removed the conversion from the A side of this GEMM; this is the
// other half of the same wave, and the caller must supply pack_tiled_half weights to match
// (`DLSS5_W16_LIN`, dlss5_launch.hpp). An f16 tile is 1024 bytes with a 32-byte row stride where
// an E4M3 tile is 512 with 16 -- the same 16x32 elements, two bytes each.
// QOUT16: the Q/K/V buffer is written as the f16 of the E4M3-rounded value rather than the E4M3
// byte, so k_window_attention's Q, K and V fragments load without converting. Exact for the same
// reason AH16 is -- every E4M3 code is representable in f16, and the value is already on that grid
// when it reaches the store. See DLSS5_QKV16 in dlss5_linalg.hpp.
template <int C, int NH, bool F8IN = false, bool HIN = false, bool VONLY = false,
          bool AH16 = DLSS5_AH16, bool W16 = false, bool QOUT16 = DLSS5_QKV16>
__global__ void k_qkv_norm_f32(const float* input, const u8* input_f8, const u8* weights,
                               u8* output, uint m, uint k, const float* scales, float qgain,
                               bool half_squares, const __half* input_h = nullptr,
                               uint head_base = 0u) {
    using namespace linalg;
    constexpr uint HEADS = C / 32u;
    constexpr uint COLS = 32u * NH;
    const uint first = blockIdx.x * 16u;
    const uint head0 = head_base + blockIdx.y * NH;
    const uint wave = threadIdx.x / 32u;
    // Single LDS buffer: the A-tile staging aliases the start of the stage
    // buffer. The tile is only read inside the K loop; stage is only
    // written after it (with a barrier between), so the aliasing is safe.
    // Cuts LDS 512+16*COLS*4 -> 16*COLS*4 (e.g. C32/NH1 2560B->2048B).
    //
    // AH16 doubles the tile, 512B -> 1KB, and costs no LDS at all: the smallest stage here is
    // NH=1's 16*32 floats = 2048 B, so the wider tile still fits inside it. Aligned because the
    // f16 path stores eight bytes at a time.
    constexpr uint ASZ = AH16 ? 2u : 1u;
    constexpr uint STAGE_COLS = VONLY ? 128u : (C == 32 && NH == 3) ? 64u : COLS;
    static_assert(16u * STAGE_COLS * 4u >= 16u * 32u * ASZ, "A-tile no longer fits inside stage");
    __shared__ __align__(16) float stage[16 * STAGE_COLS];
    u8* tile = reinterpret_cast<u8*>(stage);
    C32 acc = C32::Splat(0.f);
    const uint col = head0 * 32u + wave * 16u;
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
                // Four E4M3 codes, formed exactly as before whichever format the tile holds.
                u32 q;
                if (F8IN) {
                    q = row < m
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
                    q = u32(e4m3_byte(x)) | u32(e4m3_byte(y)) << 8 |
                        u32(e4m3_byte(z)) << 16 | u32(e4m3_byte(w)) << 24;
                } else {
                    const float4 v = row < m
                        ? *reinterpret_cast<const float4*>(input + row * k + g * 32u + c4)
                        : make_float4(0.f, 0.f, 0.f, 0.f);
                    q = u32(e4m3_byte(v.x)) | u32(e4m3_byte(v.y)) << 8 |
                        u32(e4m3_byte(v.z)) << 16 | u32(e4m3_byte(v.w)) << 24;
                }
                u8* dst = tile + (rlocal * 32u + c4) * ASZ;
                if constexpr (AH16)
                    *reinterpret_cast<uint2*>(dst) = e4m3x4_to_half4(q);
                else
                    *reinterpret_cast<u32*>(dst) = q;
            }
        }
        __syncthreads();
        // One declaration for both formats; the row stride scales with the element size. On the
        // host pass HiddenA<true> is still an fp8 fragment -- wrong, never runs, and the only
        // spelling that lets the B fragment below agree with it on element size so the launch stub
        // compiles.
        using AT = typename HiddenA<AH16>::type;
        AT a = AT::Load(tile, 0, 32u * ASZ);
        // One declaration for both weight formats, as on the A side above.
        constexpr uint BSZ = W16 ? 2u : 1u;
        using BT = typename WeightB<W16>::type;
        BT b = BT::LoadRow(weights, ((col / 16u) * (k / 32u) + g) * 512u * BSZ, 16 * BSZ);
        acc.MultiplyAccumulate(a, b);
        __syncthreads();
    }
    const uint norm_heads = head0 < 2u * HEADS
                                ? (NH < 2u * HEADS - head0 ? NH : 2u * HEADS - head0)
                                : 0u;
    const uint norm_waves = 2u * norm_heads;
    if (wave < norm_waves)
        acc.StoreF32(stage, wave * 16u * 4u, STAGE_COLS * 4u);
    __syncthreads();
    // 16 rows x normalized heads; one thread per row-head keeps the strict channel sum order.
    const uint item = threadIdx.x;
    const uint ROW_THREADS = 16 * norm_heads;
    if (item < ROW_THREADS) {
        const uint row = item / norm_heads;
        const uint headg = head0 + (item % norm_heads);
        const float* s0 = &stage[row * STAGE_COLS + (item % norm_heads) * 32u];
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
        // The eight dwords of E4M3 codes are formed identically either way; only the store differs.
        u32 w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint ch = 0; ch < 16u; ++ch)
            w[ch >> 2] |= u32(e4m3_byte(s0[ch] * inv)) << ((ch & 3u) * 8u);
        for (uint ch = 0; ch < 16u; ++ch)
            w[4u + (ch >> 2)] |= u32(e4m3_byte(s1[ch] * inv)) << ((ch & 3u) * 8u);
        // Spelled out per format rather than scaled by a constexpr 1: reassociating the pointer
        // arithmetic changed the E4M3 path's codegen (one 64-bit add folded into v_and_or_b32,
        // one instruction fewer and provably the same address) and that path is supposed to be
        // untouched. A free instruction is not worth losing the before/after identity check.
        u8* out;
        if constexpr (QOUT16)
            out = output + ((size_t)(first + row) * (3u * C) + headg * 32u) * 2u;
        else
            out = output + (size_t)(first + row) * (3u * C) + headg * 32u;
        uint4* o4 = reinterpret_cast<uint4*>(out);
        if constexpr (QOUT16) {
            // 32 codes -> 32 halves: four 16-byte stores instead of two. Rows are 64-byte aligned
            // here (3C and headg*32 are both multiples of 32, doubled), so uint4 stays legal.
            for (uint j = 0; j < 4u; ++j) {
                uint2 a = e4m3x4_to_half4(w[j * 2u]), b = e4m3x4_to_half4(w[j * 2u + 1u]);
                o4[j] = make_uint4(a.x, a.y, b.x, b.y);
            }
        } else {
            o4[0] = make_uint4(w[0], w[1], w[2], w[3]);
            o4[1] = make_uint4(w[4], w[5], w[6], w[7]);
        }
    }
    if (wave >= norm_waves) {
        const uint headg = head0 + wave / 2u;
        u8* out;
        if constexpr (QOUT16)
            out = output + ((size_t)first * (3u * C) + headg * 32u + (wave % 2u) * 16u) * 2u;
        else
            out = output + (size_t)first * (3u * C) + headg * 32u + (wave % 2u) * 16u;
        for (uint i = 0; i < acc.Length(); ++i) {
            uint2 rc = acc.GetCoordinate(i);
            if (first + rc.x < m) {
                const u8 q = e4m3_byte(acc.Get(i));
                if constexpr (QOUT16)
                    reinterpret_cast<uint16_t*>(out)[(size_t)rc.x * (3u * C) + rc.y] =
                        e4m3_to_half_u16(q);
                else
                    out[(size_t)rc.x * (3u * C) + rc.y] = q;
            }
        }
    }
}

} // namespace dlss5
