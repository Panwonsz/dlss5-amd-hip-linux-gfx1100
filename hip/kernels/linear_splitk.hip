// k_linear_f32 for `ordered == 2`, split four ways over K across the CTA's waves.
//
// ordered == 2 already defines the result as four independent K partitions, each accumulated from
// zero in K order, added to the (preset) accumulator in partition order:
//     acc = ((((R + P0) + P1) + P2) + P3)
// The original kernel computes P0..P3 one after another in every wave (k/32 serial steps, each a
// global load -> LDS stage -> barrier -> MMA -> barrier). Here wave w computes P_w for all four of
// the CTA's 16-column tiles (k/128 steps, no barriers, A read straight from global), the partials
// meet in LDS, and wave t then forms acc for column tile t in exactly the original order and runs
// the original epilogue. Same MMA sequence per (tile, partition), same f32 adds, same order:
// bit-identical. Used for the ViT contract and projection linears (640 x 1024, K = 4096 / 1024).
//
// Scope (the launcher falls back to k_linear_f32 otherwise): F8IN input, f32 output, f32 residual,
// mode != 4, !matrix_residual, m % 16 == 0, n % 64 == 0, k % 128 == 0.
#pragma once
#include "dlss5_linalg.hpp"

namespace dlss5 {

template <bool W16>
__global__ void __launch_bounds__(128) k_linear_f32_splitk4(const u8* input_f8, const u8* weights,
                                                            float* output, uint m, uint n, uint k,
                                                            const float* residual,
                                                            const float* scales, int mode) {
    using namespace linalg;
    using BT = typename WeightB<W16>::type;
    constexpr uint BSZ = W16 ? 2u : 1u;
    const uint wave = threadIdx.x / 32u, lane = threadIdx.x & 31u;
    const uint first = blockIdx.x * 16u;
    const uint col0 = blockIdx.y * 64u;
    const uint steps = k / 128u;  // K steps of 32 per partition

    // ---- phase 1: wave w computes partition w for the four column tiles --------------------------
    C32 part[4];
    for (uint t = 0; t < 4u; ++t) part[t] = C32::Splat(0.f);
    for (uint s = 0; s < steps; ++s) {
        const uint g = wave * steps + s;
        const A8 a = A8::Load(input_f8 + size_t(first) * k, g * 32u, k);
        for (uint t = 0; t < 4u; ++t) {
            const uint col = col0 + t * 16u;
            const BT b = BT::LoadRow(weights, ((col / 16u) * (k / 32u) + g) * 512u * BSZ, 16u * BSZ);
            part[t].MultiplyAccumulate(a, b);
        }
    }
    // [partition][tile][element][lane]: every wave writes and reads by (element, lane), so the
    // fragment layout is the same on both sides.
    __shared__ float stage[4 * 4 * 8 * 32];
    for (uint t = 0; t < 4u; ++t)
        for (uint i = 0; i < part[t].Length(); ++i)
            stage[((wave * 4u + t) * 8u + i) * 32u + lane] = part[t].Get(i);
    __syncthreads();

    // ---- phase 2: wave t owns column tile t, as in k_linear_f32 --------------------------------
    const uint col = col0 + wave * 16u;
    C32 acc = C32::Splat(0.f);
    if (residual)  // has_residual && !late_residual (ordered 2), matrix_residual == false
        for (uint i = 0; i < acc.Length(); ++i) {
            const uint2 rc = acc.GetCoordinate(i);
            const float rv = residual[(first + rc.x) * n + col + rc.y];
            acc.Set(i, H(rv * scales[col + rc.y]));
        }
    for (uint p = 0; p < 4u; ++p)
        for (uint i = 0; i < acc.Length(); ++i)
            acc.Set(i, acc.Get(i) + stage[((p * 4u + wave) * 8u + i) * 32u + lane]);

    // ---- epilogue: k_linear_f32's, for this variant (mode != 4, f32 output) -------------------
    for (uint i = 0; i < acc.Length(); ++i) {
        const uint2 rc = acc.GetCoordinate(i);
        float v = acc.Get(i);
        if (mode == 2) {
            v = from_e4m3(e4m3_byte(ActivatePoly(v)));
        } else if (mode == 1) {
            v = from_e4m3(e4m3_byte(H(v)));
        } else if (mode == 5) {
            v = from_e4m3(e4m3_byte(v));
        } else if (mode == 0) {
            v = H(v);
        }
        if (output) output[(first + rc.x) * n + col + rc.y] = v;
    }
}

} // namespace dlss5
