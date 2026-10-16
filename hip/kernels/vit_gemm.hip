// ViT linears as one register-blocked GEMM (hip-37, 2026-10-16). Bit-identical to the kernels they
// replace; DLSS5_VIT_GEMM=0 switches back for timing.
//
// Profile 5: the ViT linears were ~4.7 ms a frame at 22-33 TFLOPS. Each of their CTAs is 16 rows x 64
// columns, so every 16-row A tile is re-read (and, for the expand, re-converted from f32 to E4M3) by
// every column block, and every weight tile is re-read by all 40 row blocks. Here a CTA is 64 rows x
// 64 columns: wave w owns rows 16w..16w+15 and all four 16-column tiles, so per K step a wave loads one
// A fragment and four B fragments and issues eight MMAs, and the four waves read the SAME four B
// fragments (one global read, three cache hits).
//
// Exactness, per output 16x16 tile: the same A values, the same B fragments, the same
// MultiplyAccumulate sequence over K in the same order, the same epilogue -- only which wave runs it
// changes.
//  - expand (was k_linear_f32 <F8OUT, AH16, W16>, ordered 0, mode 2): A is the f16 of the E4M3 code of
//    the f32 input, which k_vit_quant_in forms once with the same e4m3_byte + e4m3x4_to_half4 the
//    linear's tile staging used. Output: the f16 of the E4M3 code (exact), which is exactly what the
//    contract's load_e4m3 decoded from the old byte buffer.
//  - contract / projection (were k_linear_f32_splitk4, ordered 2): partition p = blockIdx.z is
//    accumulated from zero in K order; k_vit_reduce then forms ((((H(r*s) + P0) + P1) + P2) + P3) and
//    runs the mode-1 epilogue, as splitk4's phase 2 does. The projection's A is the E4M3 attention
//    output read with A8::Load, the same decode as before.
//  - QKV (was k_linear_half<false>, 1 partition, ordered 0, raw 2): A = __float2half of the f32
//    contract output (k_vit_reduce writes it), genuine f16 weights, output 0.f + acc as before.
#pragma once
#include <type_traits>
#include "dlss5_linalg.hpp"

namespace dlss5 {

// f32 -> f16 of its E4M3 rounding, four values per thread.
__global__ void k_vit_quant_in(const float* in, __half* out, uint count4) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count4) return;
    const float4 v = reinterpret_cast<const float4*>(in)[i];
    const u32 q = u32(e4m3_byte(v.x)) | u32(e4m3_byte(v.y)) << 8 | u32(e4m3_byte(v.z)) << 16 |
                  u32(e4m3_byte(v.w)) << 24;
    reinterpret_cast<uint2*>(out)[i] = e4m3x4_to_half4(q);
}

// AIN: 0 = f16 A (HiddenA<true> fragments, E4M3-valued), 1 = E4M3 bytes (A8), 2 = genuine f16 A with
// genuine f16 weights (A16 / B16, the k_linear_half types).
// EPI: 0 = partition partial (f32, [z][m][n]), 1 = expand mode 2 (f16 of the E4M3 code),
//      2 = raw f32 (0.f + acc).
template <int AIN, int EPI>
__global__ void __launch_bounds__(128) k_vit_gemm(const void* a_in, const void* weights, float* out,
                                                  __half* out_h, uint m, uint n, uint k) {
    using namespace linalg;
    using AT = typename std::conditional<AIN == 2, A16,
               typename std::conditional<AIN == 1, A8, typename HiddenA<true>::type>::type>::type;
    using BT = typename std::conditional<AIN == 2, B16, BW16>::type;
    constexpr uint ASZ = AIN == 1 ? 1u : 2u;
    const uint wave = threadIdx.x / 32u;
    const uint row0 = blockIdx.x * 64u + wave * 16u;
    const uint col0 = blockIdx.y * 64u;
    const uint parts = gridDim.z;
    const uint steps = k / 32u / parts;
    const uint g0 = blockIdx.z * steps;
    C32 acc[4];
    for (uint t = 0; t < 4u; ++t) acc[t] = C32::Splat(0.f);
    const u8* abase = reinterpret_cast<const u8*>(a_in) + size_t(row0) * k * ASZ;
    for (uint s = 0; s < steps; ++s) {
        const uint g = g0 + s;
        const AT a = AT::Load(abase, g * 32u * ASZ, k * ASZ);
        for (uint t = 0; t < 4u; ++t) {
            const uint col = col0 + t * 16u;
            const BT b = BT::LoadRow(weights, ((col / 16u) * (k / 32u) + g) * 1024u, 32u);
            acc[t].MultiplyAccumulate(a, b);
        }
    }
    for (uint t = 0; t < 4u; ++t) {
        const uint col = col0 + t * 16u;
        for (uint i = 0; i < acc[t].Length(); ++i) {
            const uint2 rc = acc[t].GetCoordinate(i);
            const size_t o = size_t(row0 + rc.x) * n + col + rc.y;
            if constexpr (EPI == 0) {
                out[size_t(blockIdx.z) * m * n + o] = acc[t].Get(i);
            } else if constexpr (EPI == 1) {
                // Decoded with e4m3x4_to_half4, the function the contract's load_e4m3 used on the
                // old byte buffer, so the f16 is the same bits whichever decoder the build selects.
                const u8 q = e4m3_byte(ActivatePoly(acc[t].Get(i)));
                out_h[o] = __ushort_as_half((unsigned short)(e4m3x4_to_half4(u32(q)).x & 0xffffu));
            } else {
                float v = 0.f;
                v = v + acc[t].Get(i);
                out[o] = v;
            }
        }
    }
}

// ((((H(r*s) + P0) + P1) + P2) + P3), mode 1, as k_linear_f32_splitk4's phase 2 + epilogue.
// out_h (optional): __float2half of the f32 output -- the QKV linear's A, as k_linear_half formed it.
__global__ void k_vit_reduce(const float* parts, const float* residual, const float* scales,
                             float* out, __half* out_h, uint m, uint n) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint count = m * n;
    if (i >= count) return;
    float v = 0.f;
    if (residual) v = H(residual[i] * scales[i % n]);
    for (uint p = 0; p < 4u; ++p) v = v + parts[size_t(p) * count + i];
    v = from_e4m3(e4m3_byte(H(v)));
    out[i] = v;
    if (out_h) out_h[i] = __float2half(v);
}

} // namespace dlss5
