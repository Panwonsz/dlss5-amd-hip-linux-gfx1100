// Native HIP window-attention fast path.
// 8×8 windows, 32-channel heads, FP8 QKV in window-major [token][3C].
#pragma once
#include "dlss5_linalg.hpp"

namespace dlss5 {

__device__ inline float fast_exp_raw(float x) {
    float a = fminf(fmaxf(fmaf(x, 0.044921875f, 1.30078125f), 1.03125f), 1.5693359375f);
    unsigned short h = __half_as_ushort(__float2half_rn(a));
    unsigned short e = (unsigned short)(((unsigned(h) << 5) + 0x8000u) & 65535u);
    return __half2float(__ushort_as_half(e));
}

__device__ inline uint window_pixel(uint window, uint token, uint width) {
    return ((window / (width / 8)) * 8 + token / 8) * width + (window % (width / 8)) * 8 + token % 8;
}

// block = (window, head). 128 threads: 4 waves × 16 query tokens.
// scores rows are padded to 68 (8B-aligned): the row sum and the byte write hit
// 2-way bank conflicts instead of 32-way. Phase 2 writes the E4M3 bytes directly
// (e4m3_byte is the identity on E4M3-grid values, so the separate re-quant pass
// is numerically redundant) as 16B stores; phase 3 is pure MMAs.
// QKV16: the qkv buffer holds the f16 of the E4M3-rounded value, so Q, K and V load with one
// load_matrix_sync each rather than load_e4m3's eight conversions per lane. Those three are 82% of
// this kernel's fragment fills -- each of the four waves reads its own Q and ALL 64 K and V tokens
// -- and this kernel is 18.10 ms of a 98 ms frame (tests/bench_attention.hip).
//
// The p8 softmax buffer stays E4M3: it is the other 18%, and widening it would need a row stride of
// a multiple of 8 halves for 16-byte-aligned fragment loads, which puts the rows at least 4 banks
// apart where 68 halves puts them 2 -- trading phase 2's bank behaviour for phase 3's conversions.
// Not obviously a win, and LDS here is the occupancy limit (12,800 B allows exactly 5 blocks/CU
// against the 6 that 218 VGPRs would permit).
//
// k_qkv_norm_f32 must agree; DLSS5_QKV16 in dlss5_linalg.hpp drives both.
template <int C, bool QKV16 = DLSS5_QKV16>
__global__ void __launch_bounds__(128, 5) k_window_attention(const u8* qkv, const float* bias,
                                                             u8* output, uint width,
                                   uint height, uint packed, bool direct_f8 = false) {
    using namespace linalg;
    constexpr int HEADS = C / 32;
    constexpr int STRIDE = 3 * C;
    const uint head = blockIdx.y;
    if (head >= uint(HEADS))
        return;
    const uint nwin = (width / 8) * (height / 8);
    const uint wave = threadIdx.x / 32u;
    const uint qtok = wave * 16u;
    __shared__ __half scores[64 * 68];
    __shared__ u8 p8[64 * 64];
    for (uint window = blockIdx.x; window < nwin; window += gridDim.x) {

    // One spelling for both formats; every offset into qkv scales with the element size.
    constexpr uint QSZ = QKV16 ? 2u : 1u;
    using QT = typename HiddenA<QKV16>::type;
    using KVT = typename WeightB<QKV16>::type;

    QT q = QT::Load(qkv, ((window * 64u + qtok) * uint(STRIDE) + head * 32u) * QSZ,
                    uint(STRIDE) * QSZ);
    for (uint k0 = 0; k0 < 64; k0 += 16) {
        KVT k = KVT::LoadCol(
            qkv, ((window * 64u + k0) * uint(STRIDE) + uint(C) + head * 32u) * QSZ,
            uint(STRIDE) * QSZ);
        C32 s = Multiply(q, k);
        for (uint i = 0; i < s.Length(); i++) {
            uint2 rc = s.GetCoordinate(i);
            uint qi = qtok + rc.x, kj = k0 + rc.y;
            scores[qi * 68 + kj] = __float2half_rn(
                fast_exp_raw(s.Get(i) + bias[head * 4096u + qi * 64u + kj]));
        }
    }
    __syncthreads();

    if (threadIdx.x < 64) {
        uint qi = threadIdx.x;
        float sum = 0.f;
        // 16 x 8B loads (4 f16 each); 136B row stride keeps 8B alignment and the
        // per-step banks 2-way. Serial f32 order 0..63 is preserved.
        for (int j = 0; j < 16; j++) {
            uint2 v = *reinterpret_cast<const uint2*>(&scores[qi * 68 + j * 4]);
            const __half2* h = reinterpret_cast<const __half2*>(&v);
            sum += __half2float(__low2half(h[0]));
            sum += __half2float(__high2half(h[0]));
            sum += __half2float(__low2half(h[1]));
            sum += __half2float(__high2half(h[1]));
        }
        float inv = 1.f / fmaxf(sum, 1e-20f);
        // 4 x 16B stores per row (16 E4M3 bytes): rows are 16B-aligned, banks
        // 8-way instead of 32-way.
        uint4* row = reinterpret_cast<uint4*>(p8 + qi * 64);
        for (int q4 = 0; q4 < 4; q4++) {
            uint4 v;
            u8* b = reinterpret_cast<u8*>(&v);
            for (int t = 0; t < 16; t++)
                b[t] = e4m3_byte(__half2float(scores[qi * 68 + q4 * 16 + t]) * inv);
            row[q4] = v;
        }
    }
    __syncthreads();

    C32 av0 = C32::Splat(0.f);
    C32 av1 = C32::Splat(0.f);
    for (uint k0 = 0; k0 < 64; k0 += 32) {
        A8 p = A8::Load(p8, qtok * 64u + k0, 64);   // p8 stays E4M3, see the note above
        const uint vbase =
            ((window * 64u + k0) * uint(STRIDE) + 2u * uint(C) + head * 32u) * QSZ;
        KVT v0 = KVT::LoadRow(qkv, vbase, uint(STRIDE) * QSZ);
        KVT v1 = KVT::LoadRow(qkv, vbase + 16u * QSZ, uint(STRIDE) * QSZ);
        av0.MultiplyAccumulate(p, v0);
        av1.MultiplyAccumulate(p, v1);
    }
    for (uint i = 0; i < av0.Length(); i++) {
        uint2 rc = av0.GetCoordinate(i);
        uint dst = packed ? (window * 64u + qtok + rc.x)
                          : window_pixel(window, qtok + rc.x, width);
        output[dst * uint(C) + head * 32u + rc.y] = e4m3_byte(direct_f8 ? av0.Get(i) : H(av0.Get(i)));
        output[dst * uint(C) + head * 32u + 16u + rc.y] = e4m3_byte(direct_f8 ? av1.Get(i) : H(av1.Get(i)));
    }
    __syncthreads();
    }
}

} // namespace dlss5
