#pragma once
#include "dlss5_common.hpp"

#define DLSS5_CONFIGURED_POLICIES 1

namespace dlss5 {

void set_launch_stream(hipStream_t s);
void launch_linear_f32(const float* in, const u8* w, float* out, uint m, uint n, uint k,
                       const float* residual, const float* scales, int mode, int ordered,
                       bool matrix_residual = false);
void launch_normalize_qkv(const float* in, const float* scales, u8* out,
                          uint tokens, uint c, float qgain, bool half_squares = true);
void launch_reframe_f32(const float* src, float* dst, uint w, uint h, uint sw, uint sh,
                        uint px, uint py, uint c, int crop, int quantize);
void launch_pool_f32(const float* src, float* dst, uint w, uint h, uint c, uint ow, uint oh);
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
void launch_split_f32(const float*,const u8*,const u8*,const u8*,float*,uint tokens);
void launch_rgb_graph(const float*,float*,float*,uint w,uint h,uint ow,uint oh);
void launch_gather_f32(const float*,const int*,float*,uint n);
void launch_up_f32(const float*,const float*,const float*,float*,uint iw,uint ih,uint ow,uint oh,uint c);
void launch_post_merge_f32(const float*,const float*,const float*,float*,uint w,uint h,bool main8_low = false);
void launch_head_f32(const float*,const float*,const float*,float*,uint n);
void launch_trace_f32(const float*,uint n,float*,uint*);

} // namespace dlss5
