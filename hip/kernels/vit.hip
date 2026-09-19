// HIP port of ViT wave kernels (640 tokens × 1024, tiled E4M3).
#pragma once
#include "dlss5_linalg.hpp"

namespace dlss5 {

// Generic tiled E4M3 GEMM: [M x K] @ [N x K]^T-as-B-tiles → [M x N], 16×16 output tiles.
__global__ void k_gemm_tiled_f8(const u8* A, const u8* Btiles, u8* C, uint M, uint N, uint K,
                                int activate) {
    using namespace linalg;
    uint m0 = blockIdx.x * 16u;
    uint n0 = blockIdx.y * 16u;
    if (m0 >= M || n0 >= N)
        return;
    C32 acc = C32::Splat(0.f);
    for (uint g = 0; g < K / 32; g++) {
        A8 a = A8::Load(A, m0 * K + g * 32u, K);
        uint boff = ((n0 / 16u) * (K / 32u) + g) * 512u;
        B8 b = B8::LoadRowPacked(Btiles, boff);
        acc.MultiplyAccumulate(a, b);
    }
    if (activate) {
        for (uint i = 0; i < acc.Length(); i++)
            acc.Set(i, Activate(acc.Get(i)));
    } else {
        for (uint i = 0; i < acc.Length(); i++)
            acc.Set(i, F(H(acc.Get(i))));
    }
    acc.StoreF8(C, m0 * N + n0, N);
}

__global__ void k_vit_gather(const u8* src, const int* map, u8* dst, uint tokens, uint src_c,
                             uint dst_c) {
    uint t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tokens * dst_c)
        return;
    uint ch = t % dst_c, tok = t / dst_c;
    int idx = map[tok * dst_c + ch];
    dst[t] = idx < 0 ? 0 : src[uint(idx)];
}

} // namespace dlss5
