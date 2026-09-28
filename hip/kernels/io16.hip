// Full-resolution I/O in f16 (DLSS5_IO16, 2026-09-28).
//
// Blocks 0 and 70 end in H(v): every value of their 1920x1152x32 output raster is an exact f16, so
// float(half) reproduces the f32 raster bit for bit. The consumers below read that f16 raster (141 MB)
// instead of the f32 one (283 MB) and compute exactly what their f32 twins compute: same expressions,
// same operand values, same order. Each thread does several channels with vector loads; the math per
// element is unchanged.
//
//   k_pool_hin       = k_pool_f32 (dst == nullptr, dst_h set)   4 channels per thread
//   k_post_merge_hin = k_post_merge_f32 (out == nullptr, out_h)  8 channels per thread
//   k_head_hin       = k_head_f32                                 one pixel per thread, 32 channels
#pragma once

namespace dlss5 {

__global__ void k_pool_hin(const __half* src, __half* dst_h, uint w, uint h, uint c, uint ow,
                           uint oh) {
    const uint i4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i4 >= ow * oh * c / 4u) return;
    const uint i = i4 * 4u, ch = i % c, t = i / c, x = t % ow, y = t / ow;
    __half r[4];
    if (x >= w / 2 || y >= h / 2) {
        for (uint k = 0; k < 4; ++k) r[k] = __float2half_rn(0.f);
    } else {
        const uint base = (y * 2 * w + x * 2) * c + ch;
        __half a[4], b[4], d[4], e[4];
        *reinterpret_cast<uint2*>(a) = *reinterpret_cast<const uint2*>(src + base);
        *reinterpret_cast<uint2*>(b) = *reinterpret_cast<const uint2*>(src + base + c);
        *reinterpret_cast<uint2*>(d) = *reinterpret_cast<const uint2*>(src + base + w * c);
        *reinterpret_cast<uint2*>(e) = *reinterpret_cast<const uint2*>(src + base + (w + 1) * c);
        for (uint k = 0; k < 4; ++k) {
            float top = H(__half2float(a[k]) + __half2float(b[k]));
            float bottom = H(__half2float(d[k]) + __half2float(e[k]));
            float v = from_e4m3(e4m3_byte(H(H(top + bottom) * .25f)));
            r[k] = __float2half_rn(v);
        }
    }
    *reinterpret_cast<uint2*>(dst_h + i) = *reinterpret_cast<const uint2*>(r);
}

__global__ void k_post_merge_hin(const float* low, const __half* pre, const float* scales,
                                 __half* out_h, uint w, uint h, bool main8_low) {
    const uint i8 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i8 >= w * h * 4u) return;
    const uint i = i8 * 8u, ch0 = i % 32u, p = i / 32u,
               lp = ((p / w) / 2) * (w / 2) + (p % w) / 2;
    __half pr[8], r[8];
    *reinterpret_cast<uint4*>(pr) = *reinterpret_cast<const uint4*>(pre + i);
    float lw[8];
    *reinterpret_cast<float4*>(lw) = *reinterpret_cast<const float4*>(low + lp * 32u + ch0);
    *reinterpret_cast<float4*>(lw + 4) = *reinterpret_cast<const float4*>(low + lp * 32u + ch0 + 4u);
    for (uint k = 0; k < 8; ++k) {
        const uint ch = ch0 + k;
        float lo = main8_low ? from_e4m3(e4m3_byte(lw[k])) : lw[k];
        float v = H(H(lo * scales[ch]) + from_e4m3(e4m3_byte(__half2float(pr[k]))) * scales[32 + ch]);
        r[k] = __float2half_rn(v);
    }
    *reinterpret_cast<uint4*>(out_h + i) = *reinterpret_cast<const uint4*>(r);
}

__global__ void k_head_hin(const __half* feat_h, const float* weights, const float* color, float* out,
                           uint n) {
    const uint p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    __half fh[32];
    for (uint q = 0; q < 4; ++q)
        *reinterpret_cast<uint4*>(fh + q * 8) = *reinterpret_cast<const uint4*>(feat_h + p * 32u + q * 8u);
    float feat[32];
    for (uint c = 0; c < 32; ++c) feat[c] = __half2float(fh[c]);
    for (uint row = 0; row < 3; ++row) {
        float acc = 0;
        for (uint c = 0; c < 32; ++c) acc += feat[c] * weights[row * 32 + c];
        float base = color[p * 4 + row] * .125f - .0625f;
        float encoded = H(acc) * .03125f + base;
        out[p * 3 + row] = fminf(fmaxf(encoded * 8.f + .5f, 0.f), 1.f);
    }
}

void launch_pool_hin(const __half* src, __half* dst_h, uint w, uint h, uint c, uint ow, uint oh) {
    if (!w || !h || w % 2 || h % 2 || ow < w / 2 || oh < h / 2 || c % 4 || !src || !dst_h ||
        (const void*)src == (const void*)dst_h)
        throw std::runtime_error("pool hin extent/alias");
    const uint n4 = ow * oh * c / 4u;
    LAUNCH(k_pool_hin, (n4 + 255) / 256, 256, src, dst_h, w, h, c, ow, oh);
}
void launch_post_merge_hin(const float* low, const __half* pre, const float* scales, __half* out_h,
                           uint w, uint h, bool main8_low) {
    if (!low || !pre || !scales || !out_h || w % 2 || h % 2 || (const void*)pre == (const void*)out_h)
        throw std::runtime_error("post merge hin extent/alias");
    const uint n8 = w * h * 4u;
    LAUNCH(k_post_merge_hin, (n8 + 255) / 256, 256, low, pre, scales, out_h, w, h, main8_low);
}
void launch_head_hin(const __half* in, const float* w, const float* color, float* out, uint n) {
    if (!in || !w || !color || !out) throw std::runtime_error("head hin buffers");
    LAUNCH(k_head_hin, (n + 63) / 64, 64, in, w, color, out, n);
}

} // namespace dlss5
