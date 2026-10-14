// k_vit_attention<true> restructured: NW waves per CTA share one streamed K/V chunk, one pass.
//
// The original runs 1,280 one-wave CTAs. Each walks the 640 keys twice (scores, then P*V), and every
// 16- or 32-key step is global load -> LDS -> barrier -> math -> barrier, so the kernel is a chain of
// ~60 dependent global-memory latencies per wave.
//
// Here:
//  - NW waves (NW 16-query tiles of the same head) share each 32-key chunk of K and V, loaded once by
//    the CTA into a double buffer; the NEXT chunk's global loads are issued before this chunk's math,
//    so their latency is hidden, and there is one barrier per chunk.
//  - Scores and P*V run in the same pass. That is legal because the probabilities written for P*V are
//    e4m3(e) with e independent of the denominator (the original normalizes only after P*V).
//
// Bit-identical to k_vit_attention<true>, by construction:
//  - the score MMA per (query tile, 16-key half) is the same Multiply(q, k) on the same bytes;
//  - e, ex and the E4M3 probability are the same expressions;
//  - denom[row] is summed by one lane per row, key by key in ascending order, exactly as before;
//  - acc0/acc1 receive the same MMAs (p, v0/v1 per 32-key chunk) in the same ascending order;
//  - the epilogue is the same code.
#pragma once
#include "dlss5_linalg.hpp"

namespace dlss5 {

// F16ACT (DLSS5_F16_ACT): the probabilities as f16 (row stride 40 halves), no E4M3 rounding. NOT bit-exact.
template <uint NW, bool F16ACT = false>
__global__ void __launch_bounds__(32 * NW) k_vit_attention_mw(const u8* qkv_f8, u8* out, uint tokens,
                                                               uint dim) {
    using namespace linalg;
    const uint wave = threadIdx.x / 32u, lane = threadIdx.x & 31u;
    const uint q0 = (blockIdx.x * NW + wave) * 16u, head = blockIdx.y;
    const uint stride = 3u * dim;
    // kv[b]: 32 keys x 32 bytes of K, then 32 keys x 32 bytes of V.
    __shared__ __attribute__((aligned(16))) u8 kv[2][2048];
    __shared__ __attribute__((aligned(16))) u8 qtile[NW][16 * 32];
    __shared__ __attribute__((aligned(16))) u8 prob[NW][F16ACT ? 16 * 40 * 2 : 16 * 32];
    __shared__ float ex[NW][16 * 33];
    __shared__ float denom_s[NW][16];

    // The wave's own 16 query rows: 32 lanes x 16 bytes.
    {
        const uint r = lane / 2u, col = (lane & 1u) * 16u;
        *reinterpret_cast<uint4*>(qtile[wave] + r * 32u + col) =
            *reinterpret_cast<const uint4*>(qkv_f8 + (q0 + r) * stride + head * 32u + col);
    }
    // One 2 KB chunk = 128 segments of 16 bytes (64 of K, 64 of V).
    auto fetch = [&](uint k0, uint seg) -> uint4 {
        const uint which = seg / 64u, s = seg % 64u, key = k0 + s / 2u, col = (s & 1u) * 16u;
        return key < tokens ? *reinterpret_cast<const uint4*>(qkv_f8 + key * stride + (1u + which) * dim +
                                                             head * 32u + col)
                            : make_uint4(0u, 0u, 0u, 0u);
    };
    constexpr uint SEGS = (128u + 32u * NW - 1u) / (32u * NW);
    uint4 pre[SEGS];
    for (uint j = 0; j < SEGS; ++j) {
        const uint seg = threadIdx.x + j * 32u * NW;
        if (seg < 128u) pre[j] = fetch(0u, seg);
    }
    __syncthreads();  // qtile visible to the wave (and orders nothing else)
    const A8 q = A8::Load(qtile[wave], 0, 32);

    float denom = 0.f;  // lanes 0..15: row `lane`
    C32 acc0 = C32::Splat(0.f), acc1 = C32::Splat(0.f);
    const uint chunks = (tokens + 31u) / 32u;
    for (uint c = 0; c < chunks; ++c) {
        u8* buf = kv[c & 1u];
        for (uint j = 0; j < SEGS; ++j) {
            const uint seg = threadIdx.x + j * 32u * NW;
            if (seg < 128u) *reinterpret_cast<uint4*>(buf + seg * 16u) = pre[j];
        }
        // Makes buf visible. Double buffering: every wave finished reading this buffer (chunk c-2)
        // before it arrived at the barrier of chunk c-1, so one barrier per chunk suffices.
        __syncthreads();
        if (c + 1u < chunks)
            for (uint j = 0; j < SEGS; ++j) {
                const uint seg = threadIdx.x + j * 32u * NW;
                if (seg < 128u) pre[j] = fetch((c + 1u) * 32u, seg);
            }
        const uint k0 = c * 32u;
        for (uint half = 0; half < 2; ++half) {
            B8 k = B8::LoadCol(buf, half * 16u * 32u, 32);
            C32 score = Multiply(q, k);
            for (uint i = 0; i < score.Length(); ++i) {
                uint2 rc = score.GetCoordinate(i);
                float affine = fminf(fmaxf(fmaf(score.Get(i),
                    __half2float(__ushort_as_half(0x2dbb)), 1.708984375f), 1.439453125f),
                                     1.9775390625f);
                uint bits = __half_as_ushort(__float2half_rn(affine));
                float e = k0 + half * 16 + rc.y < tokens
                              ? __half2float(__ushort_as_half(uint16_t((bits << 4) + 0x4000u)))
                              : 0.f;
                ex[wave][rc.x * 33 + half * 16 + rc.y] = e;
                if constexpr (F16ACT)
                    reinterpret_cast<__half*>(prob[wave])[rc.x * 40 + half * 16 + rc.y] = __float2half_rn(e);
                else
                    prob[wave][rc.x * 32 + half * 16 + rc.y] = e4m3_byte(e);
            }
        }
        // ex/prob are this wave's own; a wave-level fence orders its LDS writes before the reads.
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        if (lane < 16)
            for (uint j = 0; j < 32; ++j) denom += ex[wave][lane * 33 + j];
        B8 v0 = B8::LoadRow(buf + 1024, 0, 32), v1 = B8::LoadRow(buf + 1024, 16, 32);
        if constexpr (F16ACT) {
            using AF = typename HiddenA<true>::type;
            const AF p = AF::Load(prob[wave], 0, 80);
            acc0.MultiplyAccumulate(p, v0);
            acc1.MultiplyAccumulate(p, v1);
        } else {
            A8 p = A8::Load(prob[wave], 0, 32);
            acc0.MultiplyAccumulate(p, v0);
            acc1.MultiplyAccumulate(p, v1);
        }
        // The next chunk rewrites ex/prob: all lanes must be done reading them.
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    }
    if (lane < 16) denom_s[wave][lane] = denom;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    for (uint i = 0; i < acc0.Length(); ++i) {
        uint2 rc = acc0.GetCoordinate(i);
        const float inv = 1.f / denom_s[wave][rc.x];
        out[(q0 + rc.x) * dim + head * 32 + rc.y] = e4m3_byte(H(acc0.Get(i) * inv));
        out[(q0 + rc.x) * dim + head * 32 + 16 + rc.y] = e4m3_byte(H(acc1.Get(i) * inv));
    }
}

} // namespace dlss5
