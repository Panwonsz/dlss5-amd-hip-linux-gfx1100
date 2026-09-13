// Pack / crop / 2×2 downsample / RGB reflect helpers (HLSL native_c32_reframe, native_c32_ds).
#pragma once
#include "dlss5_common.hpp"

namespace dlss5 {

__global__ void k_shift_pack_f32(const float* src, float* dst, uint w, uint h, uint sw, uint sh,
                                 uint pad_x, uint pad_y, uint c) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint n = sw * sh * c;
    if (i >= n)
        return;
    uint ch = i % c, t = i / c, x = t % sw, y = t / sw;
    int sx = int(x) - int(pad_x), sy = int(y) - int(pad_y);
    if (sx < 0 || sy < 0 || sx >= int(w) || sy >= int(h)) {
        dst[i] = 0.f;
        return;
    }
    dst[i] = src[(uint(sy) * w + uint(sx)) * c + ch];
}

__global__ void k_shift_pack_f8(const u8* src, u8* dst, uint w, uint h, uint sw, uint sh,
                                uint pad_x, uint pad_y, uint c) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint n = sw * sh * c;
    if (i >= n)
        return;
    uint ch = i % c, t = i / c, x = t % sw, y = t / sw;
    int sx = int(x) - int(pad_x), sy = int(y) - int(pad_y);
    if (sx < 0 || sy < 0 || sx >= int(w) || sy >= int(h)) {
        dst[i] = 0;
        return;
    }
    dst[i] = src[(uint(sy) * w + uint(sx)) * c + ch];
}

// Raster [H][W][C] → window-major [windows][64][C] for 8×8 Swin attention.
__global__ void k_pack_windows(const u8* src, u8* dst, uint w, uint h, uint c) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint n = w * h * c;
    if (i >= n)
        return;
    uint ch = i % c, t = i / c, x = t % w, y = t / w;
    uint window = (y / 8) * (w / 8) + (x / 8);
    uint tok = (y % 8) * 8 + (x % 8);
    dst[(window * 64u + tok) * c + ch] = src[i];
}

__global__ void k_unpack_windows(const u8* src, u8* dst, uint w, uint h, uint c) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint n = w * h * c;
    if (i >= n)
        return;
    uint ch = i % c, t = i / c, x = t % w, y = t / w;
    uint window = (y / 8) * (w / 8) + (x / 8);
    uint tok = (y % 8) * 8 + (x % 8);
    dst[i] = src[(window * 64u + tok) * c + ch];
}

__global__ void k_crop_f8(const u8* src, u8* dst, uint w, uint h, uint sw, uint sh, uint pad_x,
                          uint pad_y, uint c) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint n = w * h * c;
    if (i >= n)
        return;
    uint ch = i % c, t = i / c, x = t % w, y = t / w;
    dst[i] = src[((y + pad_y) * sw + (x + pad_x)) * c + ch];
}

// 2×2 average pool then optional copy (projection is a separate GEMM).
__global__ void k_pool2x2_f8(const u8* src, float* dst, uint w, uint h, uint c) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint ow = w / 2, oh = h / 2;
    if (i >= ow * oh * c)
        return;
    uint ch = i % c, t = i / c, x = t % ow, y = t / ow;
    float s = 0.f;
    for (int dy = 0; dy < 2; dy++)
        for (int dx = 0; dx < 2; dx++)
            s += from_e4m3(src[((y * 2 + dy) * w + (x * 2 + dx)) * c + ch]);
    dst[i] = s * 0.25f;
}

__global__ void k_upsample2x_f8(const u8* src, u8* dst, uint w, uint h, uint c) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint dw = w * 2, dh = h * 2;
    if (i >= dw * dh * c)
        return;
    uint ch = i % c, t = i / c, x = t % dw, y = t / dw;
    dst[i] = src[((y / 2) * w + (x / 2)) * c + ch];
}

__global__ void k_rgb_reflect(const float* rgb, float* out, uint w, uint h, uint ow, uint oh) {
    uint y = blockIdx.y, x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= oh || x >= ow)
        return;
    uint r = y % (2 * h - 2);
    uint sy = r < h ? r : (2 * h - 2) - r;
    uint i = (y * ow + x) * 4, j = (sy * w + (x < w ? x : w - 1)) * 4;
    out[i] = rgb[j];
    out[i + 1] = rgb[j + 1];
    out[i + 2] = rgb[j + 2];
    out[i + 3] = rgb[j + 3];
}

__global__ void k_f32_to_f8(const float* src, u8* dst, uint n) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = e4m3_byte(src[i]);
}

__global__ void k_f8_to_f32(const u8* src, float* dst, uint n) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = from_e4m3(src[i]);
}

__global__ void k_head_rgb(const u8* feat, const float* w, const float* color, float* rgb,
                           uint n, uint c, float scale) {
    uint p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n)
        return;
    float acc[3] = {0, 0, 0};
    for (uint ch = 0; ch < c; ch++) {
        float v = from_e4m3(feat[p * c + ch]);
        acc[0] += v * w[0 * c + ch];
        acc[1] += v * w[1 * c + ch];
        acc[2] += v * w[2 * c + ch];
    }
    rgb[p * 3 + 0] = color[p * 4 + 0] + acc[0] * scale;
    rgb[p * 3 + 1] = color[p * 4 + 1] + acc[1] * scale;
    rgb[p * 3 + 2] = color[p * 4 + 2] + acc[2] * scale;
}

} // namespace dlss5
