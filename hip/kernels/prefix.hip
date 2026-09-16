// Native HIP prefix stage (ALU noise + 16→32 prefix GEMM).
#pragma once
#include "dlss5_linalg.hpp"

namespace dlss5 {

__global__ void k_prefix(const float* input, const float* weights, const float* temporal_rgb,
                         float* output, uint seed, uint width, uint height, uint temporal,
                         __half* output_h = nullptr) {
    using namespace linalg;
    const uint t = threadIdx.x;
    const uint first = (blockIdx.x + blockIdx.y * 65535u) * 16u;
    if (first >= width * height)
        return;
    __shared__ __half ftile[16 * 32];
    __shared__ __half btile[2 * 512];
    __shared__ float otile[16 * 32];

    for (uint i = t; i < 1024; i += 32) {
        uint n = i / 512, k = (i % 512) / 16, j = i % 16;
        float w = k < 16 ? weights[(n * 16 + j) * 16 + k] : 0.f;
        btile[i] = __float2half(w);
    }
    if (t < 16) {
        const uint p = first + t;
        uint x = p % 8, y = (p / 8) % 8;
        uint tile = p / 64;
        x = (tile % (width / 8)) * 8 + p % 8;
        y = (tile / (width / 8)) * 8 + (p % 64) / 8;
        uint h = pcg((x * 0x8da6b343u) ^ (y * 0xd8163841u) ^ (seed * 0x9e3779b9u) ^ 0x243f6a88u);
        float a = uniform24(h * 0xcaa5b80d + 0x21dd796b);
        float b = uniform24(h * 0x2c9277b5 + 0xac564b05);
        float c = uniform24(h * 0x83232c31 + 0x3463e0ac);
        float d = uniform24(h * 0xfa6dc5f9 + 0x4712a88e);
        float r0 = sqrtf(-2.f * logf(a)), r1 = sqrtf(-2.f * logf(b));
        constexpr float twopi = 6.283185482025146f;
        float g0 = H(r0 * cosf(twopi * c));
        float g1 = H(r1 * cosf(twopi * d));
        float g2 = H(r1 * sinf(twopi * d));
        float r = H(H(H(input[p * 4]) - 0.5f) * 0.125f);
        float g = H(H(H(input[p * 4 + 1]) - 0.5f) * 0.125f);
        float bl = H(H(H(input[p * 4 + 2]) - 0.5f) * 0.125f);
        float features[16] = {g1, g2, g, bl, g0, 1, 0.0078125f, 1, r, g, 1, 1, bl, r, 1, 0};
        if (temporal) {
            float hx = temporal_rgb[(y * width + x) * 4 + 0];
            float hy = temporal_rgb[(y * width + x) * 4 + 1];
            float hz = temporal_rgb[(y * width + x) * 4 + 2];
            features[13] = H(H(H(hx) - 0.5f) * 0.125f);
            features[2] = H(H(H(hy) - 0.5f) * 0.125f);
            features[3] = H(H(H(hz) - 0.5f) * 0.125f);
        }
        for (uint i = 0; i < 16; i++)
            ftile[t * 32 + i] = __float2half(features[i]);
        for (uint i = 16; i < 32; i++)
            ftile[t * 32 + i] = __float2half(0.f);
    }
    __syncthreads();
    A16 fa = A16::Load(ftile, 0, 32 * 2);
    for (uint n = 0; n < 2; n++) {
        B16 wb = B16::LoadRow(btile, n * 512 * 2, 16 * 2);
        C32 z = Multiply(fa, wb);
        for (uint i = 0; i < z.Length(); i++)
            z.Set(i, H(z.Get(i)));
        z.StoreF32(otile, n * 16 * 4, 32 * 4);
    }
    __syncthreads();
    for (uint i = 0; i < 16; i++) {
        float v = otile[i * 32 + t];
        output[first * 32 + i * 32 + t] = v;
        if (output_h)
            output_h[first * 32 + i * 32 + t] = __float2half_rn(v);
    }
}

} // namespace dlss5
