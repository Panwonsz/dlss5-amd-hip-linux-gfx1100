#pragma once
#include <cstdlib>
#include "dlss5_common.hpp"

#define DLSS5_CONFIGURED_POLICIES 1

namespace dlss5 {

void set_launch_stream(hipStream_t s);
void launch_linear_f32(const float* in, const u8* w, float* out, uint m, uint n, uint k,
                        const float* residual, const float* scales, int mode, int ordered,
                        bool matrix_residual = false);
void launch_linear_f32_f8out(const float* in, const u8* w, float* out, u8* out_f8, uint m,
                             uint n, uint k, const float* residual, const float* scales, int mode,
                             int ordered, bool matrix_residual = false);
void launch_linear_f32_f8in(const u8* in, const u8* w, float* out, uint m, uint n, uint k,
                            const float* residual, const float* scales, int mode, int ordered,
                            bool matrix_residual = false);
void launch_linear_f32_f8in_hrid(const u8* in, const u8* w, float* out, uint m, uint n, uint k,
                                 const __half* residual_h, const float* scales, int mode, int ordered,
                                 bool matrix_residual = false);
void launch_linear_f32_f8in_hrid_hout(const u8* in, const u8* w, __half* out_h, uint m, uint n,
                                       uint k, const __half* residual_h, const float* scales,
                                       int mode, int ordered, bool matrix_residual = false);
void launch_linear_f32_f8in_hrid_hout_raster(const u8* in, const u8* w, __half* out_h,
                                              __half* out_h_raster, uint m, uint n, uint k,
                                              const __half* residual_h, const float* scales, int mode,
                                              int ordered, bool matrix_residual, uint rw, uint rh,
                                              uint rsw, uint rpx, uint rpy);

void launch_linear_f32_f8in_f8out(const u8* in, const u8* w, float* out, u8* out_f8, uint m,
                                   uint n, uint k, const float* residual, const float* scales,
                                   int mode, int ordered, bool matrix_residual = false);
void launch_linear_f32_f8in_f8out_f8res(const u8* in, const u8* w, float* out, u8* out_f8, uint m,
                                         uint n, uint k, const u8* residual_f8,
                                         const float* scales, int mode, int ordered,
                                         bool matrix_residual);
void launch_linear_f32_f8in_f8out_hrid(const u8* in, const u8* w, float* out, u8* out_f8, uint m,
                                       uint n, uint k, const __half* residual_h,
                                       const float* scales, int mode, int ordered,
                                       bool matrix_residual);
void launch_linear_f32_f8in_f8out_raster(const u8* in, const u8* w, float* out, u8* out_f8_raster,
                                         uint m, uint n, uint k, const float* residual,
                                         const float* scales, int mode, int ordered,
                                         bool matrix_residual, uint rw, uint rh, uint rsw,
                                         uint rpx, uint rpy);

// DLSS5_W16: the FFN's weights are uploaded AND read as f16 instead of E4M3.
//
// Measured on a 7900 XT, same kernel and bit-identical output: -44.6% at C=32 rising to -53.5% at
// C=256. RDNA3 has no FP8 matrix instruction, so load_e4m3 fills each weight fragment with eight
// strided single-byte reads and eight conversions where the f16 path issues one vectorised
// load_matrix_sync. The absolute saving is constant across C (~0.69 ms/call), which is what identifies
// it as per-load overhead rather than anything to do with bandwidth or arithmetic.
//
// This ONE function drives both the upload in network.hip and the dispatch in kernels.hip, and it has to
// stay that way. If the two ever disagree, E4M3 bytes are read as f16 -- a stable, plausible, entirely
// wrong picture -- and the network's only self-check is replay equality, which such a mismatch passes
// without complaint. Compare hip-network70's `mean` and `mean_abs_change` across the two settings
// instead: the f16 weights are the dequantised E4M3 values, so those must match digit for digit.
inline bool dlss5_w16() {
    static const bool on = [] {
        const char* asked = std::getenv("DLSS5_W16");
        return asked != nullptr && std::atoi(asked) != 0;
    }();
    return on;
}

// The 2x2 of weight format (w16) x hidden-activation format (a16), for benchmarks. Not for the network:
// w16 requires f16-packed weights from the caller, exactly as launch_ffn_f32_w16 does.
void launch_ffn_f32_bench(int c, const float* in, const u8* w, const float* scales, float* out,
                          uint tokens, bool chain_residual, bool precise_c32, bool w16, bool a16);

// The FFN with f16 weights, for measuring what the E4M3 fragment load costs. The caller packs them.
void launch_ffn_f32_w16(int c, const float* in, const u8* w16, const float* scales, float* out,
                        uint tokens, bool chain_residual = false, bool precise_c32 = false);
void launch_ffn_f32_hout(int c, const float* in, const u8* w, const float* scales, __half* out_h,
                          uint tokens, bool chain_residual, bool precise_c32);
void launch_ffn_f32_hin_hout(int c, const __half* in_h, const u8* w, const float* scales,
                             __half* out_h, uint tokens, bool chain_residual, bool precise_c32);
void launch_ffn_f32_f8(int c, const float* in, const u8* w, const float* scales, float* out_f32,
                        u8* out_f8, uint tokens, bool chain_residual, bool precise_c32);
void launch_ffn_f32_f8in_f8out(int c, const u8* in_f8, const u8* w, const float* scales,
                               float* out_f32, u8* out_f8, uint tokens, bool chain_residual,
                               bool precise_c32);
void launch_ffn_f32_hin_f8out(int c, const __half* in_h, const u8* w, const float* scales,
                              float* out_f32, u8* out_f8, uint tokens, bool chain_residual,
                              bool precise_c32);
void launch_qkv_norm_f32_f8in(int c, const u8* in, const u8* w, u8* out, uint m,
                              const float* scales, float qgain, bool half_squares);
void launch_qkv_norm_f32_hin(int c, const __half* in, const u8* w, u8* out, uint m,
                             const float* scales, float qgain, bool half_squares);
void launch_split_f32_f8(const float* in, const u8* pre, const u8* ex, const u8* ct, float* out,
                          u8* out_f8, uint tokens);
void launch_split_f32_f8in_f8(const u8* in_f8, const u8* pre, const u8* ex, const u8* ct,
                              float* out, u8* out_f8, uint tokens);
void launch_split_f32_hin_f8(const __half* in_h, const u8* pre, const u8* ex, const u8* ct,
                             float* out, u8* out_f8, uint tokens);
void launch_normalize_qkv(const float* in, const float* scales, u8* out,
                          uint tokens, uint c, float qgain, bool half_squares = true);
void launch_qkv_norm_f32(int c, const float* in, const u8* w, u8* out, uint m,
                         const float* scales, float qgain, bool half_squares);
void launch_reframe_f32(const float* src, float* dst, uint w, uint h, uint sw, uint sh,
                        uint px, uint py, uint c, int crop, int quantize);
void launch_reframe_f32_hin(const __half* src_h, float* dst, uint w, uint h, uint sw, uint sh,
                             uint px, uint py, uint c, int crop, int quantize);
void launch_reframe_f8in_f8out(const u8* src, u8* dst, uint w, uint h, uint sw, uint sh,
                               uint px, uint py, uint c);
void launch_reframe_f32_to_h(const float* src, __half* dst, uint w, uint h, uint sw, uint sh,
                             uint px, uint py, uint c);
void launch_reframe_hin_hout(const __half* src, __half* dst, uint w, uint h, uint sw, uint sh,
                             uint px, uint py, uint c);
void launch_pool_f32(const float* src, float* dst, uint w, uint h, uint c, uint ow, uint oh);
void launch_pool_f32_h(const float* src, float* dst, __half* dst_h, uint w, uint h, uint c,
                       uint ow, uint oh);
void launch_ffn_f32(int c, const float* in, const u8* w, const float* scales, float* out, uint tokens,
                    bool chain_residual = false, bool precise_c32 = false);

void launch_ffn(int c, const u8* in, const u8* w, u8* out, uint tokens);
void launch_qkv(int c, const u8* in, const u8* w, u8* out, uint tokens);
void launch_project(int c, const u8* in, const u8* w, const float* scales, u8* out, uint tokens,
                    const u8* residual = nullptr);
void launch_attention(int c, const u8* qkv, const float* bias, u8* out, uint w, uint h,
                      uint packed = 0, bool direct_f8 = false);
void launch_prefix(const float* in, const float* w, const float* temporal, float* out, uint seed,
                   uint width, uint height, uint temporal_on);
void launch_prefix_h(const float* in, const float* w, const float* temporal, float* out,
                     __half* out_h, uint seed, uint width, uint height, uint temporal_on);
void launch_gemm_tiled(const u8* a, const u8* b, u8* c, uint M, uint N, uint K, int activate);
void launch_f32_to_f8(const float* s, u8* d, uint n);
void launch_f8_to_f32(const u8* s, float* d, uint n);
void launch_pool2x2(const u8* s, float* d, uint w, uint h, uint c);
void launch_upsample2x(const u8* s, u8* d, uint w, uint h, uint c);
void launch_rgb_reflect(const float* rgb, float* out, uint w, uint h, uint ow, uint oh);
void launch_head_rgb(const u8* feat, const float* w, const float* color, float* rgb, uint n, uint c,
                     float scale);
void launch_vit_gather(const u8* src, const int* map, u8* dst, uint tokens, uint src_c, uint dst_c);
void launch_ds(const u8* s, const u8* w, u8* d, uint wdt, uint h, uint cin, uint cout);
void launch_add_f8(const u8* a, const u8* b, u8* d, uint n);
void launch_upsample_add(const u8* lo, const u8* skip, u8* d, uint w, uint h, uint c);
void launch_linear_skip(const u8* in, const u8* w, const u8* skip, u8* out, uint tokens, uint cin,
                        uint cout);
void launch_split_ffwd(const u8* in, const u8* pre, const u8* expand, const u8* contract, u8* out,
                       uint tokens);
void launch_vit_attention(const float* qkv, u8* out, uint tokens, uint dim);
void launch_vit_attention_f8(const u8* qkv_f8, u8* out, uint tokens, uint dim);
void launch_post70_merge(const u8* main, const u8* skip, const float* scales, u8* dst, uint n,
                         uint c);
void launch_shift_pack_f32(const float* s, float* d, uint w, uint h, uint sw, uint sh, uint px,
                           uint py, uint c);
void launch_shift_pack_f8(const u8* s, u8* d, uint w, uint h, uint sw, uint sh, uint px, uint py,
                          uint c);
void launch_crop_f8(const u8* s, u8* d, uint w, uint h, uint sw, uint sh, uint px, uint py, uint c);
void launch_pack_windows(const u8* s, u8* d, uint w, uint h, uint c);
void launch_unpack_windows(const u8* s, u8* d, uint w, uint h, uint c);
void launch_residual_scale(const u8* gemm, const u8* skip, const float* scale, u8* dst, uint n,
                           uint c);
void launch_decoder_scatter(const u8* lo, const u8* skip, const float* scale, u8* dst, uint in_w,
                            uint in_h, uint out_w, uint out_h, uint c);
void launch_fused_mh(int c, const u8* in, const u8* ffn, const u8* proj0, const float* proj0_s,
                     const u8* qkv, const float* bias, const u8* proj1, const float* proj1_s,
                     u8* qkv_tmp, u8* attn_tmp, u8* out, uint w, uint h, int do_proj0,
                     int packed = 0);

hipStream_t get_launch_stream();
void launch_linear_half(const float*,const __half*,float*,uint m,uint n,uint k,uint partitions,int ordered,int raw);
void launch_linear_half_h(const float*,const __half*,float*,__half*,uint m,uint n,uint k,uint partitions,int ordered,int raw);
void launch_linear_half_hin(const __half*,const __half*,float*,__half*,uint m,uint n,uint k,uint partitions,int ordered,int raw);
void launch_split_f32(const float*,const u8*,const u8*,const u8*,float*,uint tokens);
void launch_rgb_graph(const float*,float*,float*,uint w,uint h,uint ow,uint oh);
void launch_gather_f32(const float*,const int*,float*,uint n);
void launch_up_f32(const float*,const float*,const float*,float*,uint iw,uint ih,uint ow,uint oh,uint c);
void launch_up_f32_h(const float*,const float*,const float*,float*,__half*,uint iw,uint ih,uint ow,uint oh,uint c);
void launch_post_merge_f32(const float*,const float*,const float*,float*,uint w,uint h,bool main8_low = false);
void launch_post_merge_f32_h(const float*,const float*,const float*,float*,__half*,uint w,uint h,bool main8_low);
void launch_head_f32(const float*,const float*,const float*,float*,uint n);
void launch_trace_f32(const float*,uint n,float*,uint*);

} // namespace dlss5
