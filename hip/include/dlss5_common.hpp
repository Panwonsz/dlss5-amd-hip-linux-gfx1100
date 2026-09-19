// Numerics for the native HIP fast chain.
// H() = f16 RNE, F() = E4M3FN RNE (OCP, bias 7, max 448).
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <cstdint>
#include <cstddef>
#include <cmath>
#ifndef __HIP_DEVICE_COMPILE__
#include <cstring>
#endif

// RDNA 3 targets: no FP8 wave-matrix ops or hardware E4M3 conversion.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
#define DLSS5_GFX11 1
#else
#define DLSS5_GFX11 0
#endif

namespace dlss5 {

using u32 = uint32_t;
using u8  = uint8_t;

__host__ __device__ inline u32 as_u32(float v) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __float_as_uint(v);
#else
    u32 u;
    std::memcpy(&u, &v, 4);
    return u;
#endif
}
__host__ __device__ inline float as_f32(u32 u) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __uint_as_float(u);
#else
    float v;
    std::memcpy(&v, &u, 4);
    return v;
#endif
}

// Software f16 RNE (HLSL bit recipe, used when NATIVE_HW_H=0).
__host__ __device__ inline float H_sw(float v) {
    u32 b = as_u32(v), sg = b & 0x80000000u, a = b & 0x7fffffffu;
    if (a >= 0x7f800000u)
        return v;
    if (a < 0x38800000u) {
        float q = nearbyintf(fabsf(v) * 16777216.0f) * 5.9604644775390625e-8f;
        return sg ? -q : q;
    }
    u32 r = (a + 0xfffu + ((a >> 13) & 1u)) & 0xffffe000u;
    return as_f32(sg | (r >= 0x47800000u ? 0x7f800000u : r));
}

#if DLSS5_GFX11
// ---- gfx11 branch-free numerics -------------------------------------------------
// gfx11 has no hardware E4M3 conversion. Branchy scalar code in wave kernels does
// not vectorize across lanes (tests/probe_frag.hip), so these integer/mask versions
// replace the software paths in device code. All three were verified bit-identical
// to the originals on the host for all 2^32 float inputs (e4m3_byte, F) and all
// 256 codes (from_e4m3).
__device__ inline u32 dlss5_mask(bool c) { return 0u - u32(c); }
// Branch-free E4M3 byte -> f16 (exact for every code; 0x7f/0xff -> f16 NaN).
// Integer-only so the 32 lanes vectorize; a private f16[256] + load_matrix_sync
// was ~19x slower (tests/probe_frag.hip: 129 vs 2471 GFLOPS).
__device__ inline __half e4m3_to_half(u32 b) {
    u32 e = (b >> 3) & 15u, m = b & 7u, s = (b & 0x80u) << 8;
    u32 normal = ((e + 8u) << 10) | (m << 7);
    u32 sub = (((0x88887760u >> (m * 4u)) & 15u) << 10) | (((0xE480u >> (m * 2u)) & 3u) << 8);
    u32 mask = 0u - u32(e != 0u);
    u32 nan = 0u - u32((b & 0x7fu) == 0x7fu);
    return __ushort_as_half(uint16_t(s | (normal & mask) | (sub & ~mask) | (nan & 0x7e00u)));
}

__device__ inline u8 e4m3_byte_nb(float v) {
    u32 b = as_u32(v), a = b & 0x7fffffffu, sg = (b >> 24) & 0x80u;
    u32 q = u32(nearbyintf(fminf(fabsf(v), 1.f) * 512.f));
    u32 r = (a + 0x7ffffu + ((a >> 20) & 1u)) & 0xfff00000u;
    u32 cn = ((r >> 23) - 120u) * 8u + ((r >> 20) & 7u);
    u32 mc = dlss5_mask(cn > 126u);
    cn = (cn & ~mc) | (126u & mc);
    u32 ms = dlss5_mask(a < 0x3c800000u), mt = dlss5_mask(a >= 0x43e00000u),
        mn = dlss5_mask(a > 0x7f800000u);
    u32 code = (q & ms) | (cn & ~ms);
    code = (code & ~mt) | (126u & mt);
    code = (code & ~mn) | (127u & mn);
    return u8(sg | code);
}
__device__ inline float F_nb(float v) {
    u32 bits = as_u32(v), a = bits & 0x7fffffffu, s = bits & 0x80000000u;
    u32 sub = as_u32(nearbyintf(fminf(fabsf(v), 1.f) * 512.f) / 512.f);
    u32 r = (a + 0x7ffffu + ((a >> 20) & 1u)) & 0xfff00000u;
    u32 mr = dlss5_mask(r > 0x43e00000u);
    r = (r & ~mr) | (0x43e00000u & mr); // clamp to 448
    u32 ms = dlss5_mask(a < 0x3c800000u), mt = dlss5_mask(a >= 0x43e00000u),
        mi = dlss5_mask(a >= 0x7f800000u);
    u32 mag = (sub & ms) | (r & ~ms);
    mag = (mag & ~mt) | (0x43e00000u & mt);
    u32 out = s | mag;
    out = (out & ~mi) | (bits & mi); // inf/NaN pass through
    return as_f32(out);
}
#endif

// Software E4M3FN RNE (HLSL Ffast / NativeFastFp8).
__host__ __device__ inline float F_sw(float v) {
    u32 bits = as_u32(v), a = bits & 0x7fffffffu;
    if (a >= 0x7f800000u)
        return v;
    float sg = v < 0 ? -1.f : 1.f;
    if (a < 0x3c800000u)
        return copysignf(nearbyintf(fabsf(v) * 512.f) / 512.f, v);
    if (a >= 0x43e00000u)
        return sg * 448.f;
    u32 r = (a + 0x7ffffu + ((a >> 20) & 1u)) & 0xfff00000u;
    float m = as_f32(r);
    return sg * (m > 448.f ? 448.f : m);
}

__host__ __device__ inline float H_hw(float v) {
// __float2half_rn verified bit-identical to H_sw on gfx1100 too
// (tests/probe_gfx11.hip section 4: 2^20 samples incl. ties, 0 mismatches).
#if defined(__HIP_DEVICE_COMPILE__)
    return __half2float(__float2half_rn(v));
#else
    return H_sw(v);
#endif
}
__host__ __device__ inline float F_hw(float v) {
#if defined(__HIP_DEVICE_COMPILE__) && DLSS5_GFX11
    return F_nb(v);
#elif defined(__HIP_DEVICE_COMPILE__)
    return float(__hip_fp8_e4m3(v));
#else
    return F_sw(v);
#endif
}

#ifndef DLSS5_HW_H
#define DLSS5_HW_H 1
#endif
#if DLSS5_HW_H
#define H H_hw
#define F F_hw
#else
#define H H_sw
#define F F_sw
#endif

__host__ __device__ inline float ActivatePoly(float v) {
    float g = fminf(fmaxf(v, -4.f), 4.f);
    return v * (g * (fabsf(g) * (-0.055908203125f) + 0.447265625f) + 0.89453125f);
}
__host__ __device__ inline float Activate(float v) { return F(ActivatePoly(v)); }

// native_c32_ffn_fused.hlsli precise q/p: no multiply-add contraction.
// Keep the legacy polynomial for other shader families and public operators.
__host__ __device__ inline float ActivatePolyC32(float v) {
    float g = fminf(fmaxf(v, -4.f), 4.f);
    // Opaque f32 multiplies keep each rounding separate without the volatile
    // private-memory round-trips that spilled 20 B/lane of scratch on gfx12.
#if defined(__HIP_DEVICE_COMPILE__)
    float qmul, pmul;
    asm("v_mul_f32_e32 %0, %1, %2" : "=v"(qmul) : "v"(fabsf(g)), "v"(-0.055908203125f));
    float q = qmul + 0.447265625f;
    asm("v_mul_f32_e32 %0, %1, %2" : "=v"(pmul) : "v"(g), "v"(q));
    float p = pmul + 0.89453125f;
    return v * p;
#else
    volatile float qmul = fabsf(g) * (-0.055908203125f);
    volatile float q = qmul + 0.447265625f;
    volatile float pmul = g * q;
    volatile float p = pmul + 0.89453125f;
    return v * p;
#endif
}

// Saturating E4M3FN round-to-nearest-even, including exponent carries.
// Preserve NaNs instead of silently treating corrupt data as a finite value.
// On gfx12 the hardware E4M3 conversion (cvt_pk_fp8_f32 after an fmed3f
// clamp to +/-448) is bit-identical to the software recipe for every
// finite input (verified: all 256 codes, all RNE midpoints, denormal scan
// and 4M random values, hip/tests/test_e4m3_hw.hip); the software path
// stays as the non-finite fallback.
__host__ __device__ inline u8 e4m3_byte(float v) {
#if defined(__HIP_DEVICE_COMPILE__) && DLSS5_GFX11
    return e4m3_byte_nb(v);
#endif
#if defined(__HIP_DEVICE_COMPILE__) && !DLSS5_GFX11
    if ((as_u32(v) & 0x7fffffffu) < 0x7f800000u)
        return u8(__builtin_amdgcn_cvt_pk_fp8_f32(__builtin_amdgcn_fmed3f(v, 448.f, -448.f), 0.f, 0, false) & 0xffu);
#endif
    u32 b = as_u32(v), a = b & 0x7fffffffu;
    u8 sg = u8((b >> 24) & 0x80u);
    if (a > 0x7f800000u)
        return u8(sg | 0x7f);
    if (a >= 0x43e00000u)
        return u8(sg | 0x7e);
    if (a < 0x3c800000u)
        return u8(sg | u8(nearbyintf(fabsf(v) * 512.f)));
    u32 rounded = (a + 0x7ffffu + ((a >> 20) & 1u)) & 0xfff00000u;
    u32 code = ((rounded >> 23) - 120u) * 8u + ((rounded >> 20) & 7u);
    return u8(sg | (code > 126u ? 126u : code));
}

__host__ __device__ inline float from_e4m3(u8 b) {
#if defined(__HIP_DEVICE_COMPILE__) && DLSS5_GFX11
    return __half2float(e4m3_to_half(b));
#endif
    u32 e = (b >> 3) & 15u, m = b & 7u;
    if (e == 15u && m == 7u)
        return as_f32(0x7fc00000u | (u32(b & 0x80u) << 24));
    float v = e ? as_f32(((e + 120u) << 23) | (m << 20)) : float(m) * 0.001953125f;
    return (b & 0x80u) ? -v : v;
}

__host__ __device__ inline u32 pcg(u32 s) {
    u32 w = ((s >> ((s >> 28) + 4)) ^ s) * 0x108ef2d9u;
    return (w >> 22) ^ w;
}
__host__ __device__ inline float uniform24(u32 s) {
    u32 w = ((s >> ((s >> 28) + 4)) ^ s) * 0x108ef2d9u;
    return float(((w >> 30) ^ (w >> 8)) + 1) * 5.9604644775390625e-8f;
}

// 512-byte B-tile packing: [K=32][N=16] row-major from row-major [N][K].
inline void pack_tiled_e4m3(u8* dst, const float* src, size_t N, size_t K) {
    for (size_t t = 0; t < N / 16; t++)
        for (size_t g = 0; g < K / 32; g++)
            for (size_t k = 0; k < 32; k++)
                for (size_t j = 0; j < 16; j++)
                    // j * 32 + k, not k * 16 + j. Same 512-byte tile in the same place, so nothing
                    // that indexes tiles changes; but a fragment lane's eight K-values for one column
                    // are now ADJACENT, which lets MatrixB::LoadRowPacked read them as two dwords
                    // instead of eight byte loads. RDNA3 has no FP8 matrix instruction, so every one of
                    // these fragments is filled by hand and the layout is what decides the cost.
                    //
                    // The f16 weight tiles (pack_tiled_half, half_matrix) keep k * 16 + j: those are
                    // read by rocwmma::load_matrix_sync, which defines its own layout.
                    dst[(t * (K / 32) + g) * 512 + j * 32 + k] =
                        e4m3_byte(src[(t * 16 + j) * K + g * 32 + k]);
}

} // namespace dlss5
