// The daemon's pixel conversion, on the GPU (DLSS5 staged entry, 2026-09-29).
//
// dlss5_run_staged takes the game's staged proxy exactly as the daemon receives it -- 1920x1080 in its
// own DXGI format, rows `pitch` bytes apart -- and must hand back exactly the bytes the daemon would.
// These kernels are linux/dlss5_nr_daemon.cpp's DecodeRow / EncodeRow and their helpers, line for line:
//  - the half / float11 / float10 packing is integer code, ported as written;
//  - the only float arithmetic is one division per UNORM channel (decode) and saturate * scale + 0.5
//    (encode). The daemon is built with g++ -O2 for x86-64: correctly rounded division, no FMA. Here
//    the division is __fdiv_rn (correctly rounded), and the encode kernel turns contraction off, so
//    saturate * scale + 0.5 is a multiply then an add, as on the CPU (checked in the ISA);
//  - the float -> integer conversions truncate values that are always in range, as on the CPU.
// Non-finite checks mirror dlss5_run's all_finite over the model's input and its whole output.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

namespace dlss5 {
namespace staged {

enum : uint32_t {
    kR32G32B32A32_FLOAT = 2,
    kR16G16B16A16_FLOAT = 10,
    kR10G10B10A2_UNORM = 24,
    kR11G11B10_FLOAT = 26,
    kR8G8B8A8_UNORM = 28,
    kR8G8B8A8_UNORM_SRGB = 29,
    kB8G8R8A8_UNORM = 87,
    kB8G8R8A8_UNORM_SRGB = 91,
};

__host__ __device__ inline uint32_t bytes_per_pixel(uint32_t format) {
    switch (format) {
    case kR32G32B32A32_FLOAT: return 16;
    case kR16G16B16A16_FLOAT: return 8;
    case kR10G10B10A2_UNORM:
    case kR11G11B10_FLOAT:
    case kR8G8B8A8_UNORM:
    case kR8G8B8A8_UNORM_SRGB:
    case kB8G8R8A8_UNORM:
    case kB8G8R8A8_UNORM_SRGB: return 4;
    default: return 0;
    }
}

__device__ inline float half_to_float(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1Fu;
    uint32_t mantissa = h & 0x3FFu;
    if (exponent == 0) {
        if (mantissa == 0) return __uint_as_float(sign);
        while ((mantissa & 0x400u) == 0) {
            mantissa <<= 1;
            exponent--;
        }
        exponent++;
        mantissa &= 0x3FFu;
    } else if (exponent == 31) {
        return __uint_as_float(sign | 0x7F800000u | (mantissa << 13));
    }
    return __uint_as_float(sign | ((exponent + 112u) << 23) | (mantissa << 13));
}

__device__ inline uint16_t float_to_half(float f) {
    const uint32_t bits = __float_as_uint(f);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t rawExponent = (bits >> 23) & 0xFFu;
    const uint32_t mantissa = bits & 0x7FFFFFu;
    if (rawExponent == 0xFFu) return uint16_t(sign | 0x7C00u | (mantissa != 0 ? 0x200u : 0u));
    const int32_t exponent = int32_t(rawExponent) - 112;
    if (exponent >= 0x1F) return uint16_t(sign | 0x7BFFu);
    if (exponent <= 0) {
        if (exponent < -10) return uint16_t(sign);
        const uint32_t withImplied = mantissa | 0x800000u;
        const uint32_t shift = uint32_t(14 - exponent);
        const uint32_t truncated = withImplied >> shift;
        const uint32_t half = 1u << (shift - 1);
        const uint32_t remainder = withImplied & ((1u << shift) - 1u);
        return uint16_t(sign + truncated +
                        ((remainder > half || (remainder == half && (truncated & 1u) != 0)) ? 1u : 0u));
    }
    const uint32_t packed = (uint32_t(exponent) << 10) | (mantissa >> 13);
    const uint32_t remainder = mantissa & 0x1FFFu;
    const uint32_t rounded =
        packed + ((remainder > 0x1000u || (remainder == 0x1000u && (packed & 1u) != 0)) ? 1u : 0u);
    return uint16_t(sign | (rounded >= 0x7C00u ? 0x7BFFu : rounded));
}

__device__ inline float float11_to_float(uint32_t v) {
    const uint32_t exponent = (v >> 6) & 0x1Fu, mantissa = v & 0x3Fu;
    return __uint_as_float(exponent == 0 ? 0u : ((exponent + 112u) << 23) | (mantissa << 17));
}

__device__ inline uint32_t float_to_float11(float f) {
    if (!(f > 0.0f)) return 0;
    const uint32_t bits = __float_as_uint(f);
    const int32_t exponent = int32_t((bits >> 23) & 0xFFu) - 112;
    if (exponent <= 0) return 0;
    if (exponent >= 0x1F) return 0x7BFu;
    return (uint32_t(exponent) << 6) | ((bits >> 17) & 0x3Fu);
}

__device__ inline float float10_to_float(uint32_t v) {
    const uint32_t exponent = (v >> 5) & 0x1Fu, mantissa = v & 0x1Fu;
    return __uint_as_float(exponent == 0 ? 0u : ((exponent + 112u) << 23) | (mantissa << 18));
}

__device__ inline uint32_t float_to_float10(float f) {
    if (!(f > 0.0f)) return 0;
    const uint32_t bits = __float_as_uint(f);
    const int32_t exponent = int32_t((bits >> 23) & 0xFFu) - 112;
    if (exponent <= 0) return 0;
    if (exponent >= 0x1F) return 0x3DFu;
    return (uint32_t(exponent) << 5) | ((bits >> 18) & 0x1Fu);
}

__device__ inline float saturate(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

__device__ inline bool finite(float v) { return (__float_as_uint(v) & 0x7F800000u) != 0x7F800000u; }

// One thread per pixel of row blockIdx.y. dst: packed RGBA f32 (the dlss5_run input layout).
__global__ void k_staged_decode(const uint8_t* src, float4* dst, uint32_t format, uint32_t pitch,
                                uint32_t width, uint32_t* bad) {
#pragma clang fp contract(off)
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
    if (x >= width) return;
    const uint8_t* row = src + size_t(y) * pitch;
    float r = 0.f, g = 0.f, b = 0.f;
    switch (format) {
    case kR16G16B16A16_FLOAT: {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(row) + size_t(x) * 4;
        r = half_to_float(p[0]);
        g = half_to_float(p[1]);
        b = half_to_float(p[2]);
        break;
    }
    case kR32G32B32A32_FLOAT: {
        const float* p = reinterpret_cast<const float*>(row) + size_t(x) * 4;
        r = p[0];
        g = p[1];
        b = p[2];
        break;
    }
    case kR8G8B8A8_UNORM:
    case kR8G8B8A8_UNORM_SRGB:
    case kB8G8R8A8_UNORM:
    case kB8G8R8A8_UNORM_SRGB: {
        const bool bgr = format == kB8G8R8A8_UNORM || format == kB8G8R8A8_UNORM_SRGB;
        r = __fdiv_rn(float(row[x * 4 + (bgr ? 2 : 0)]), 255.0f);
        g = __fdiv_rn(float(row[x * 4 + 1]), 255.0f);
        b = __fdiv_rn(float(row[x * 4 + (bgr ? 0 : 2)]), 255.0f);
        break;
    }
    case kR10G10B10A2_UNORM: {
        const uint32_t v = reinterpret_cast<const uint32_t*>(row)[x];
        r = __fdiv_rn(float(v & 0x3FFu), 1023.0f);
        g = __fdiv_rn(float((v >> 10) & 0x3FFu), 1023.0f);
        b = __fdiv_rn(float((v >> 20) & 0x3FFu), 1023.0f);
        break;
    }
    case kR11G11B10_FLOAT: {
        const uint32_t v = reinterpret_cast<const uint32_t*>(row)[x];
        r = float11_to_float(v & 0x7FFu);
        g = float11_to_float((v >> 11) & 0x7FFu);
        b = float10_to_float((v >> 22) & 0x3FFu);
        break;
    }
    default: break;
    }
    dst[size_t(y) * width + x] = make_float4(r, g, b, 1.0f);
    if (!finite(r) || !finite(g) || !finite(b)) atomicOr(bad, 1u);
}

// src: the model's RGB f32 output, rows `width` pixels apart (the first 1080 rows of its 1152).
// Writes the pixel bytes of each row; the caller has already copied the input's bytes (row padding
// included) into dst, which is what the daemon sends back.
__global__ void k_staged_encode(const float* src, uint8_t* dst, uint32_t format, uint32_t pitch,
                                uint32_t width) {
    // HIP compiles with -ffp-contract=fast, which fuses saturate * scale + 0.5 into one FMA (seen in
    // the ISA; __fmul_rn/__fadd_rn do not prevent it, they are inlined plain * and +) -- one rounding
    // instead of the daemon's two. Off here: v_mul_f32 then v_add_f32.
#pragma clang fp contract(off)
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
    if (x >= width) return;
    const float* s = src + (size_t(y) * width + x) * 3;
    uint8_t* row = dst + size_t(y) * pitch;
    switch (format) {
    case kR16G16B16A16_FLOAT: {
        uint16_t* p = reinterpret_cast<uint16_t*>(row) + size_t(x) * 4;
        p[0] = float_to_half(s[0]);
        p[1] = float_to_half(s[1]);
        p[2] = float_to_half(s[2]);
        p[3] = float_to_half(1.0f);
        break;
    }
    case kR32G32B32A32_FLOAT: {
        float* p = reinterpret_cast<float*>(row) + size_t(x) * 4;
        p[0] = s[0];
        p[1] = s[1];
        p[2] = s[2];
        p[3] = 1.0f;
        break;
    }
    case kR8G8B8A8_UNORM:
    case kR8G8B8A8_UNORM_SRGB:
    case kB8G8R8A8_UNORM:
    case kB8G8R8A8_UNORM_SRGB: {
        const bool bgr = format == kB8G8R8A8_UNORM || format == kB8G8R8A8_UNORM_SRGB;
        const uint8_t r = uint8_t(saturate(s[0]) * 255.0f + 0.5f);
        const uint8_t g = uint8_t(saturate(s[1]) * 255.0f + 0.5f);
        const uint8_t b = uint8_t(saturate(s[2]) * 255.0f + 0.5f);
        row[x * 4 + (bgr ? 2 : 0)] = r;
        row[x * 4 + 1] = g;
        row[x * 4 + (bgr ? 0 : 2)] = b;
        row[x * 4 + 3] = 255;
        break;
    }
    case kR10G10B10A2_UNORM: {
        const uint32_t r = uint32_t(saturate(s[0]) * 1023.0f + 0.5f);
        const uint32_t g = uint32_t(saturate(s[1]) * 1023.0f + 0.5f);
        const uint32_t b = uint32_t(saturate(s[2]) * 1023.0f + 0.5f);
        reinterpret_cast<uint32_t*>(row)[x] = r | (g << 10) | (b << 20) | (3u << 30);
        break;
    }
    case kR11G11B10_FLOAT:
        reinterpret_cast<uint32_t*>(row)[x] = float_to_float11(s[0]) | (float_to_float11(s[1]) << 11) |
                                              (float_to_float10(s[2]) << 22);
        break;
    default: break;
    }
}

__global__ void k_staged_finite(const float* p, size_t n, uint32_t* bad) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        if (!finite(p[i])) {
            atomicOr(bad, 1u);
            return;
        }
}

} // namespace staged
} // namespace dlss5
