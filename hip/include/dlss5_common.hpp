// Numerics for the native HIP fast chain.
// H() = f16 RNE, F() = E4M3FN RNE (OCP, bias 7, max 448).
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <cstdint>
#include <cstdlib>
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
__device__ inline uint16_t e4m3_to_half_bits(u32 b) {
    u32 e = (b >> 3) & 15u, m = b & 7u, s = (b & 0x80u) << 8;
    u32 normal = ((e + 8u) << 10) | (m << 7);
    u32 sub = (((0x88887760u >> (m * 4u)) & 15u) << 10) | (((0xE480u >> (m * 2u)) & 3u) << 8);
    u32 mask = 0u - u32(e != 0u);
    u32 nan = 0u - u32((b & 0x7fu) == 0x7fu);
    return uint16_t(s | (normal & mask) | (sub & ~mask) | (nan & 0x7e00u));
}
__device__ inline __half e4m3_to_half(u32 b) {
    return __ushort_as_half(e4m3_to_half_bits(b));
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

// Four packed E4M3 codes -> the four f16 that decode them, packed the same way.
//
// This is the store side of the AH16 trick, and the same argument as MatrixC::StoreQuantF16: the
// value is ALREADY on the E4M3 grid when it reaches here, and every E4M3 code is exactly
// representable in f16, so widening it changes no number the matrix unit sees. What it changes is
// where the conversion happens -- once per element as the A-tile is staged into LDS, instead of
// eight times per lane on every fragment fill, for every wave that reads the tile.
//
// Takes and returns packed words rather than elements because both callers already have their four
// codes in one dword and write them with one store.
__device__ inline uint2 e4m3x4_to_half4(u32 q) {
#if DLSS5_GFX11
    return make_uint2(u32(e4m3_to_half_bits(q & 0xffu)) |
                          (u32(e4m3_to_half_bits((q >> 8) & 0xffu)) << 16),
                      u32(e4m3_to_half_bits((q >> 16) & 0xffu)) |
                          (u32(e4m3_to_half_bits(q >> 24)) << 16));
#else
    // Host-pass branch. e4m3_to_half_bits is gfx11-only (that architecture has no hardware E4M3
    // conversion, so the software one lives under #if DLSS5_GFX11), and HIP parses every __global__
    // body on the host too in order to emit its launch stub. Nothing here ever runs; it only has to
    // compile. Third time this trap has cost a build -- see BW16, HiddenA, StoreQuantF16.
    return make_uint2(q, q);
#endif
}

// DLSS5_QKV16 lives here rather than in dlss5_linalg.hpp because it sizes a host-side
// allocation (Scratch::q8) as well as selecting a fragment type, and network.hip does not
// include the linalg header. Same reason dlss5_w16() sits beside its two packers.
//
// DLSS5_QKV16: k_qkv_norm_f32 writes the Q/K/V scratch buffer as the f16 of the E4M3-rounded value,
// and k_window_attention reads it that way -- so its Q, K and V fragments load with one
// load_matrix_sync each instead of load_e4m3's per-lane unpack. The same AH16 argument as the FFN's
// hidden buffer and the linear/qkv A-tiles, applied across two kernels rather than within one.
//
// 82% of attention's fragment fills come from that buffer (Q once per wave, K and V all 64 tokens
// per wave, four waves), and attention is 18.10 ms of a 98 ms frame. The other 18% is the p8 LDS
// buffer, which is worth less and costs bank conflicts to align -- see the plan document.
//
// **Default OFF**, unlike AH16. The risk here is not occupancy but bandwidth: K and V are each read
// four times over, so doubling the buffer doubles the traffic on a kernel already at roughly 30% of
// the card's bandwidth, and whether that outweighs removing the conversions is a question for the
// GPU. Build with -DDLSS5_QKV16=1 to measure; flip this default once it has a number, exactly as
// DLSS5_W16 was flipped.
//
// It costs the q8 scratch: n*3 bytes becomes n*6, ~215 MB -> ~430 MB. The ViT shares that
// allocation but writes and reads it with its own kernel pair (k_normalize_qkv, k_vit_attention_f8)
// and never crosses formats with the window path, so only the size matters there.
#ifndef DLSS5_QKV16
#define DLSS5_QKV16 0
#endif

// Scalar sibling of e4m3x4_to_half4, for call sites that have one code rather than four packed.
// Same host branch, same reason: e4m3_to_half_bits lives under #if DLSS5_GFX11 and HIP parses every
// __global__ body on the host too.
__device__ inline uint16_t e4m3_to_half_u16(u32 b) {
#if DLSS5_GFX11
    return e4m3_to_half_bits(b);
#else
    return uint16_t(b);
#endif
}

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
                    dst[(t * (K / 32) + g) * 512 + k * 16 + j] =
                        e4m3_byte(src[(t * 16 + j) * K + g * 32 + k]);
}

// The same tiles in f16: 1024 bytes with a 32-byte row stride, and deliberately ROUND-TRIPPED
// through E4M3 first.
//
// Not full f16 precision. The point is a performance change with a provably identical picture, so
// the matrix unit must see the values it already saw. __float2half(src[...]) would be slightly MORE
// accurate and would change the output, which forfeits the only check available -- hip-network70's
// statistics matching across the two settings.
inline void pack_tiled_half(__half* dst, const float* src, size_t N, size_t K) {
    for (size_t t = 0; t < N / 16; t++)
        for (size_t g = 0; g < K / 32; g++)
            for (size_t k = 0; k < 32; k++)
                for (size_t j = 0; j < 16; j++)
                    dst[(t * (K / 32) + g) * 512 + k * 16 + j] =
                        __float2half(from_e4m3(e4m3_byte(src[(t * 16 + j) * K + g * 32 + k])));
}

// DLSS5_W16: weights are uploaded AND read as f16 instead of E4M3, everywhere the four hot kernels
// read them -- the FFN's `fw`, `k_linear_f32`'s `p0`, `p1` and the ViT's `ex`/`ct`/`pw`, and
// `k_qkv_norm_f32`'s `qw`. Seven uploads, eighteen launchers.
//
// **On by default**, because it is worth 55 ms of 153 and the alternative default is a machine that
// silently runs 60% slow when someone forgets to export a variable. `DLSS5_W16=0` is the escape
// hatch; it costs ~250 MB of VRAM, which is the reason the escape hatch exists at all.
//
// RDNA3 has no FP8 matrix instruction, so load_e4m3 fills each weight fragment with eight strided
// single-byte reads and eight conversions where the f16 path issues one vectorised
// load_matrix_sync. The absolute saving is constant across C (~0.69 ms/call on the FFN), which is
// what identifies it as per-load overhead rather than bandwidth or arithmetic.
//
// This ONE function drives every upload and every dispatch, and it has to stay that way. If any two
// of them disagree, E4M3 bytes are read as f16 -- a stable, plausible, entirely wrong picture -- and
// the network's only self-check is replay equality, which such a mismatch passes without complaint.
// It lives here, beside the two packers it chooses between, so that the choice and the formats
// cannot drift into separate headers. The gate is hip-network70's `mean` and `mean_abs_change`:
// pack_tiled_half round-trips through E4M3, so the f16 weights are the dequantised E4M3 values and
// every digit must match across the two settings.
//
// Anything that packs its own weights must ask this before choosing a packer -- see
// upload_tiled_weights() in dlss5_runtime.hpp, which is the supported way to do it.
inline bool dlss5_w16() {
    static const bool on = [] {
        const char* v = std::getenv("DLSS5_W16");
        // ONLY the exact string "0" turns it off. Unset, empty, or anything unparseable leaves the
        // fast path on. The asymmetry is deliberate now that the default is on: `atoi` would read
        // "off", "no" and "flase" as zero and silently cost 55 ms, and a mistyped opt-OUT that
        // quietly does nothing is far cheaper to notice than a mistyped opt-out that works.
        return !(v != nullptr && v[0] == '0' && v[1] == '\0');
    }();
    return on;
}

} // namespace dlss5
