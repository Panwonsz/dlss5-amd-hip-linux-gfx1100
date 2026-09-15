#include "dlss5_runtime.hpp"
#include "dlss5_launch.hpp"
#include "../kernels/ffn_fused.hip"
#include "../kernels/qkv.hip"
#include "../kernels/qkv_norm.hip"
#include "../kernels/project.hip"
#include "../kernels/prefix.hip"
#include "../kernels/attention.hip"
#include "../kernels/layout.hip"
#include "../kernels/vit.hip"
#include "../kernels/stage.hip"
#include "../kernels/fused.hip"
#include "../kernels/ops.hip"

namespace dlss5 {

static thread_local hipStream_t g_stream;

void set_launch_stream(hipStream_t s) { g_stream = s; }

#define LAUNCH(name, grid, block, ...)                                                             \
    do {                                                                                           \
        name<<<grid, block, 0, g_stream>>>(__VA_ARGS__);                                            \
    } while (0)

static uint grid_cap(uint n) { return n ? n : 1u; }

void launch_ffn_f32(int c, const float* in, const u8* w, const float* scales, float* out, uint tokens,
                    bool chain_residual, bool precise_c32) {
    if (!tokens || in == out || (c == 32 && !scales)) throw std::runtime_error("FFN extent/alias/scales");
    uint groups = (tokens + 15) / 16;
    switch (c) {
    case 32: LAUNCH((k_ffn_f32<32, false>), groups, 64, in, w, scales, out, nullptr, tokens,
                    chain_residual, precise_c32); break;
    case 64: LAUNCH((k_ffn_f32<64, false>), groups, 128, in, w, scales, out, nullptr, tokens); break;
    case 128: LAUNCH((k_ffn_f32<128, false>), groups, 256, in, w, scales, out, nullptr, tokens); break;
    case 256: LAUNCH((k_ffn_f32<256, false>), groups, 512, in, w, scales, out, nullptr, tokens); break;
    default: throw std::runtime_error("FFN channels");
    }
}
void launch_ffn_f32_f8(int c, const float* in, const u8* w, const float* scales, float* out_f32,
                       u8* out_f8, uint tokens, bool chain_residual, bool precise_c32) {
    if (!tokens || c == 32 || !out_f32 || !out_f8)
        throw std::runtime_error("FFN f8 extent/alias");
    uint groups = (tokens + 15) / 16;
    switch (c) {
    case 64: LAUNCH((k_ffn_f32<64, true>), groups, 128, in, w, scales, out_f32, out_f8, tokens); break;
    case 128: LAUNCH((k_ffn_f32<128, true>), groups, 256, in, w, scales, out_f32, out_f8, tokens); break;
    case 256: LAUNCH((k_ffn_f32<256, true>), groups, 512, in, w, scales, out_f32, out_f8, tokens); break;
    default: throw std::runtime_error("FFN f8 channels");
    }
}

void launch_normalize_qkv(const float* in, const float* scales, u8* out,
                          uint tokens, uint c, float qgain, bool half_squares) {
    if (!tokens || !c || c % 32 || !scales) throw std::runtime_error("QKV extent/scales");
    uint count = tokens * (c / 32) * 3;
    LAUNCH(k_normalize_qkv, (count + 255) / 256, 256, in, scales, out, tokens, c, qgain, half_squares);
}

// GEMM + normalize + quantize in one launch (bit-identical to the split pair).
void launch_qkv_norm_f32(int c, const float* in, const u8* w, u8* out, uint m,
                         const float* scales, float qgain, bool half_squares) {
    if (!m || !c || c % 32 || !scales) throw std::runtime_error("QKV norm extent/scales");
    switch (c) {
    case 32:
        LAUNCH((k_qkv_norm_f32<32, 1, false>), dim3((m + 15) / 16, 3u), 64, in, nullptr, w, out, m,
               c, scales, qgain, half_squares);
        break;
    case 64:
        LAUNCH((k_qkv_norm_f32<64, 2, false>), dim3((m + 15) / 16, 3u * c / 64u), 128, in, nullptr,
               w, out, m, c, scales, qgain, half_squares);
        break;
    case 128:
        LAUNCH((k_qkv_norm_f32<128, 2, false>), dim3((m + 15) / 16, 3u * c / 64u), 128, in, nullptr,
               w, out, m, c, scales, qgain, half_squares);
        break;
    case 256:
        LAUNCH((k_qkv_norm_f32<256, 2, false>), dim3((m + 15) / 16, 3u * c / 64u), 128, in, nullptr,
               w, out, m, c, scales, qgain, half_squares);
        break;
    case 512:
        LAUNCH((k_qkv_norm_f32<512, 2, false>), dim3((m + 15) / 16, 3u * c / 64u), 128, in, nullptr,
               w, out, m, c, scales, qgain, half_squares);
        break;
    default:
        throw std::runtime_error("QKV norm extent");
    }
}
void launch_qkv_norm_f32_f8in(int c, const u8* in, const u8* w, u8* out, uint m,
                              const float* scales, float qgain, bool half_squares) {
    if (!m || c == 32 || !in || !out || !scales)
        throw std::runtime_error("QKV norm f8 extent");
    switch (c) {
    case 64:
        LAUNCH((k_qkv_norm_f32<64, 2, true>), dim3((m + 15) / 16, 3u * c / 64u), 128, nullptr, in,
               w, out, m, c, scales, qgain, half_squares);
        break;
    case 128:
        LAUNCH((k_qkv_norm_f32<128, 2, true>), dim3((m + 15) / 16, 3u * c / 64u), 128, nullptr, in,
               w, out, m, c, scales, qgain, half_squares);
        break;
    case 256:
        LAUNCH((k_qkv_norm_f32<256, 2, true>), dim3((m + 15) / 16, 3u * c / 64u), 128, nullptr, in,
               w, out, m, c, scales, qgain, half_squares);
        break;
    case 512:
        LAUNCH((k_qkv_norm_f32<512, 2, true>), dim3((m + 15) / 16, 3u * c / 64u), 128, nullptr, in,
               w, out, m, c, scales, qgain, half_squares);
        break;
    default:
        throw std::runtime_error("QKV norm f8 channels");
    }
}
void launch_reframe_f32(const float* src, float* dst, uint w, uint h, uint sw, uint sh,
                        uint px, uint py, uint c, int crop, int quantize) {
    if (!w || !h || sw % 8 || sh % 8 || w + px > sw || h + py > sh || src == dst)
        throw std::runtime_error("reframe extent/alias");
    uint count = (crop ? w * h : sw * sh) * c;
    LAUNCH(k_reframe_f32, (count / 4 + 255) / 256, 256, src, dst, w, h, sw, sh, px, py, c, crop, quantize);
}
void launch_pool_f32(const float* src, float* dst, uint w, uint h, uint c, uint ow, uint oh) {
    if (!w || !h || w % 2 || h % 2 || ow < w / 2 || oh < h / 2 || src == dst)
        throw std::runtime_error("pool extent/alias");
    LAUNCH(k_pool_f32, (ow * oh * c + 255) / 256, 256, src, dst, w, h, c, ow, oh);
}

void launch_linear_f32(const float* in, const u8* w, float* out, uint m, uint n, uint k,
                       const float* residual, const float* scales, int mode, int ordered, bool matrix_residual) {
    if (!m || !n || !k || n % 16 || k % 32 || mode < 0 || mode > 5 ||
        (residual && !scales) || in == out || ordered < 0 || ordered > 2 ||
        (ordered == 2 && k % 128))
        throw std::runtime_error("linear geometry/alias");
    LAUNCH((k_linear_f32<false, false>), dim3((m + 15) / 16, (n + 63) / 64), 128, in, nullptr, w, out,
           nullptr, m, n, k, residual, scales, mode, ordered, matrix_residual);
}
void launch_linear_f32_f8in(const u8* in, const u8* w, float* out, uint m, uint n, uint k,
                            const float* residual, const float* scales, int mode, int ordered,
                            bool matrix_residual) {
    if (!m || !n || !k || n % 16 || k % 32 || mode < 0 || mode > 5 ||
        (residual && !scales) || ordered < 0 || ordered > 2 || (ordered == 2 && k % 128) ||
        reinterpret_cast<const void*>(in) == reinterpret_cast<const void*>(out))
        throw std::runtime_error("linear geometry/alias");
    LAUNCH((k_linear_f32<true, false>), dim3((m + 15) / 16, (n + 63) / 64), 128, nullptr, in, w, out,
           nullptr, m, n, k, residual, scales, mode, ordered, matrix_residual);
}
void launch_linear_f32_f8in_f8out(const u8* in, const u8* w, float* out, u8* out_f8, uint m,
                                  uint n, uint k, const float* residual, const float* scales,
                                  int mode, int ordered, bool matrix_residual) {
    if (!m || !n || !k || n % 16 || k % 32 || mode < 0 || mode > 5 ||
        (residual && !scales) || ordered < 0 || ordered > 2 || (ordered == 2 && k % 128) ||
        reinterpret_cast<const void*>(in) == reinterpret_cast<const void*>(out) || !out_f8)
        throw std::runtime_error("linear geometry/alias");
    LAUNCH((k_linear_f32<true, true>), dim3((m + 15) / 16, (n + 63) / 64), 128, nullptr, in, w, out,
           out_f8, m, n, k, residual, scales, mode, ordered, matrix_residual);
}

void launch_ffn(int c, const u8* in, const u8* w, u8* out, uint tokens) {
    uint groups = grid_cap(tokens / 16);
    dim3 block(32 * (c / 16));
    switch (c) {
    case 32:
        LAUNCH(k_ffn_fused<32>, groups, block, in, w, nullptr, out, tokens, 1, 1);
        break;
    case 64:
        LAUNCH(k_ffn_fused<64>, groups, block, in, w, nullptr, out, tokens, 1, 1);
        break;
    case 128:
        LAUNCH(k_ffn_fused<128>, groups, block, in, w, nullptr, out, tokens, 1, 1);
        break;
    case 256:
        LAUNCH(k_ffn_fused<256>, groups, block, in, w, nullptr, out, tokens, 1, 1);
        break;
    case 512:
        LAUNCH(k_ffn_fused<512>, groups, block, in, w, nullptr, out, tokens, 1, 1);
        break;
    default:
        throw std::runtime_error("ffn channels");
    }
}

void launch_qkv(int c, const u8* in, const u8* w, u8* out, uint tokens) {
    dim3 grid(grid_cap(tokens / 16), (3 * c + 63) / 64);
    switch (c) {
    case 32:
        LAUNCH(k_qkv<32>, grid, 32, in, w, out, tokens, 1);
        break;
    case 64:
        LAUNCH(k_qkv<64>, grid, 32, in, w, out, tokens, 1);
        break;
    case 128:
        LAUNCH(k_qkv<128>, grid, 32, in, w, out, tokens, 1);
        break;
    case 256:
        LAUNCH(k_qkv<256>, grid, 32, in, w, out, tokens, 1);
        break;
    case 512:
        LAUNCH(k_qkv<512>, grid, 32, in, w, out, tokens, 1);
        break;
    default:
        throw std::runtime_error("qkv channels");
    }
}

void launch_project(int c, const u8* in, const u8* w, const float* scales, u8* out, uint tokens,
                    const u8* residual) {
    if (out == in)
        throw std::runtime_error("projection input/output alias");
    if (!residual) residual = in;
    dim3 grid(grid_cap(tokens / 16), c / 64);
    switch (c) {
    case 32:
        LAUNCH((k_project<32, 2>), dim3(grid_cap(tokens / 16), c / 32), 32, in, w, scales, out, tokens,
               1, residual);
        break;
    case 64:
        LAUNCH((k_project<64, 4>), grid, 32, in, w, scales, out, tokens, 1, residual);
        break;
    case 128:
        LAUNCH((k_project<128, 4>), grid, 32, in, w, scales, out, tokens, 1, residual);
        break;
    case 256:
        LAUNCH((k_project<256, 4>), grid, 32, in, w, scales, out, tokens, 1, residual);
        break;
    case 512:
        LAUNCH((k_project<512, 4>), grid, 32, in, w, scales, out, tokens, 1, residual);
        break;
    default:
        throw std::runtime_error("project channels");
    }
}

void launch_attention(int c, const u8* qkv, const float* bias, u8* out, uint w, uint h,
                      uint packed, bool direct_f8) {
    dim3 grid(grid_cap((w / 8) * (h / 8)), c / 32);
    switch (c) {
    case 32:
        LAUNCH(k_window_attention<32>, grid, 128, qkv, bias, out, w, h, packed, direct_f8);
        break;
    case 64:
        LAUNCH(k_window_attention<64>, grid, 128, qkv, bias, out, w, h, packed, direct_f8);
        break;
    case 128:
        LAUNCH(k_window_attention<128>, grid, 128, qkv, bias, out, w, h, packed, direct_f8);
        break;
    case 256:
        LAUNCH(k_window_attention<256>, grid, 128, qkv, bias, out, w, h, packed, direct_f8);
        break;
    case 512:
        LAUNCH(k_window_attention<512>, grid, 128, qkv, bias, out, w, h, packed, direct_f8);
        break;
    default:
        throw std::runtime_error("attn channels");
    }
}

void launch_prefix(const float* in, const float* w, const float* temporal, float* out, uint seed,
                   uint width, uint height, uint temporal_on) {
    uint tokens = width * height;
    LAUNCH(k_prefix, (tokens + 15) / 16, 32, in, w, temporal, out, seed, width, height, temporal_on);
}

void launch_gemm_tiled(const u8* a, const u8* b, u8* c, uint M, uint N, uint K, int activate) {
    LAUNCH(k_gemm_tiled_f8, dim3(M / 16, N / 16), 32, a, b, c, M, N, K, activate);
}

void launch_f32_to_f8(const float* s, u8* d, uint n) {
    LAUNCH(k_f32_to_f8, (n + 255) / 256, 256, s, d, n);
}
void launch_f8_to_f32(const u8* s, float* d, uint n) {
    LAUNCH(k_f8_to_f32, (n / 4 + 255) / 256, 256, s, d, n);
}
void launch_pool2x2(const u8* s, float* d, uint w, uint h, uint c) {
    uint n = (w / 2) * (h / 2) * c;
    LAUNCH(k_pool2x2_f8, (n + 255) / 256, 256, s, d, w, h, c);
}
void launch_upsample2x(const u8* s, u8* d, uint w, uint h, uint c) {
    uint n = (w * 2) * (h * 2) * c;
    LAUNCH(k_upsample2x_f8, (n + 255) / 256, 256, s, d, w, h, c);
}
void launch_rgb_reflect(const float* rgb, float* out, uint w, uint h, uint ow, uint oh) {
    LAUNCH(k_rgb_reflect, dim3((ow + 31) / 32, oh), 32, rgb, out, w, h, ow, oh);
}
void launch_head_rgb(const u8* feat, const float* w, const float* color, float* rgb, uint n, uint c,
                     float scale) {
    LAUNCH(k_head_rgb, (n + 63) / 64, 64, feat, w, color, rgb, n, c, scale);
}
void launch_vit_gather(const u8* src, const int* map, u8* dst, uint tokens, uint src_c, uint dst_c) {
    uint n = tokens * dst_c;
    LAUNCH(k_vit_gather, (n + 255) / 256, 256, src, map, dst, tokens, src_c, dst_c);
}
void launch_ds(const u8* s, const u8* w, u8* d, uint wdt, uint h, uint cin, uint cout) {
    uint n = (wdt / 2) * (h / 2) * cout;
    LAUNCH(k_ds_project, (n + 255) / 256, 256, s, w, d, wdt, h, cin, cout);
}
void launch_add_f8(const u8* a, const u8* b, u8* d, uint n) {
    LAUNCH(k_add_f8, (n + 255) / 256, 256, a, b, d, n);
}
void launch_upsample_add(const u8* lo, const u8* skip, u8* d, uint w, uint h, uint c) {
    uint n = (w * 2) * (h * 2) * c;
    LAUNCH(k_upsample_add_f8, (n + 255) / 256, 256, lo, skip, d, w, h, c);
}
void launch_linear_skip(const u8* in, const u8* w, const u8* skip, u8* out, uint tokens, uint cin,
                        uint cout) {
    uint n = tokens * cout;
    LAUNCH(k_linear_skip, (n + 255) / 256, 256, in, w, skip, out, tokens, cin, cout);
}
void launch_split_ffwd(const u8* in, const u8* pre, const u8* expand, const u8* contract, u8* out,
                       uint tokens) {
    LAUNCH(k_split_ffwd, tokens / 16, 32, in, pre, expand, contract, out, tokens);
}
void launch_vit_attention(const float* qkv, u8* out, uint tokens, uint dim) {
    if (tokens % 16 || dim % 32)
        throw std::runtime_error("vit attention extent");
    LAUNCH(k_vit_attention, dim3(tokens / 16, dim / 32), 32, qkv, out, tokens, dim);
}
void launch_post70_merge(const u8* main, const u8* skip, const float* scales, u8* dst, uint n,
                         uint c) {
    LAUNCH(k_post70_merge, (n * c + 255) / 256, 256, main, skip, scales, dst, n, c);
}
void launch_shift_pack_f32(const float* s, float* d, uint w, uint h, uint sw, uint sh, uint px,
                           uint py, uint c) {
    uint n = sw * sh * c;
    LAUNCH(k_shift_pack_f32, (n + 255) / 256, 256, s, d, w, h, sw, sh, px, py, c);
}
void launch_shift_pack_f8(const u8* s, u8* d, uint w, uint h, uint sw, uint sh, uint px, uint py,
                          uint c) {
    uint n = sw * sh * c;
    LAUNCH(k_shift_pack_f8, (n + 255) / 256, 256, s, d, w, h, sw, sh, px, py, c);
}
void launch_crop_f8(const u8* s, u8* d, uint w, uint h, uint sw, uint sh, uint px, uint py, uint c) {
    uint n = w * h * c;
    LAUNCH(k_crop_f8, (n + 255) / 256, 256, s, d, w, h, sw, sh, px, py, c);
}
void launch_pack_windows(const u8* s, u8* d, uint w, uint h, uint c) {
    uint n = w * h * c;
    LAUNCH(k_pack_windows, (n + 255) / 256, 256, s, d, w, h, c);
}
void launch_unpack_windows(const u8* s, u8* d, uint w, uint h, uint c) {
    uint n = w * h * c;
    LAUNCH(k_unpack_windows, (n + 255) / 256, 256, s, d, w, h, c);
}
void launch_residual_scale(const u8* gemm, const u8* skip, const float* scale, u8* dst, uint n,
                           uint c) {
    LAUNCH(k_residual_scale, (n * c + 255) / 256, 256, gemm, skip, scale, dst, n, c);
}
void launch_decoder_scatter(const u8* lo, const u8* skip, const float* scale, u8* dst, uint in_w,
                            uint in_h, uint out_w, uint out_h, uint c) {
    uint n = out_w * out_h * c;
    LAUNCH(k_decoder_scatter, (n + 255) / 256, 256, lo, skip, scale, dst, in_w, in_h, out_w, out_h,
           c);
}

void launch_fused_mh(int c, const u8* in, const u8* ffn, const u8* proj0, const float* proj0_s,
                     const u8* qkv, const float* bias, const u8* proj1, const float* proj1_s,
                     u8* qkv_tmp, u8* attn_tmp, u8* out, uint w, uint h, int do_proj0, int packed) {
    // Branch activations and residual features are DIFFERENT tensors. Never
    // overwrite a matrix input while another output-channel CTA still reads it.
    if (!packed) throw std::runtime_error("Swin requires explicit window packing");
    const uint tokens = w * h;
    launch_ffn(c, in, ffn, attn_tmp, tokens);
    if (do_proj0)
        launch_project(c, attn_tmp, proj0, proj0_s, out, tokens, in);
    else
        throw std::runtime_error("C32 requires its learned FFN residual coefficients");
    launch_qkv(c, out, qkv, qkv_tmp, tokens);
    launch_attention(c, qkv_tmp, bias, attn_tmp, w, h, 1);
    launch_project(c, attn_tmp, proj1, proj1_s, out, tokens, out);
}

hipStream_t get_launch_stream() { return g_stream; }
void launch_linear_half(const float* in,const __half* w,float* out,uint m,uint n,uint k,uint parts,int ordered,int raw){
 if(!m||!n||!k||n%16||k%32||!parts||(k/32)%parts||in==out||raw<0||raw>2)throw std::runtime_error("half linear geometry/alias");
 LAUNCH(k_linear_half,dim3((m+15)/16,n/16),32,in,w,out,m,n,k,parts,ordered,raw);
}
void launch_split_f32(const float* in,const u8* pre,const u8* ex,const u8* ct,float* out,uint n){
  if(!n||in==out)throw std::runtime_error("split geometry/alias");LAUNCH((k_split_f32<false>),dim3((n+15)/16,8),128,in,pre,ex,ct,out,nullptr,n);
}
void launch_split_f32_f8(const float* in,const u8* pre,const u8* ex,const u8* ct,float* out,u8* out_f8,uint n){
  if(!n||in==out||!out_f8)throw std::runtime_error("split f8 geometry/alias");LAUNCH((k_split_f32<true>),dim3((n+15)/16,8),128,in,pre,ex,ct,out,out_f8,n);
}
void launch_rgb_graph(const float* in,float* raster,float* tiles,uint w,uint h,uint ow,uint oh){
 if(w<2||h<2||ow%8||oh%8||ow<w||oh<h)throw std::runtime_error("RGB geometry");LAUNCH(k_rgb_graph,(ow*oh+255)/256,256,in,raster,tiles,w,h,ow,oh);
}
void launch_gather_f32(const float* in,const int* map,float* out,uint n){if(in==out)throw std::runtime_error("gather alias");LAUNCH(k_gather_f32,(n+255)/256,256,in,map,out,n);}
void launch_up_f32(const float* lo,const float* skip,const float* scale,float* out,uint iw,uint ih,uint ow,uint oh,uint c){
 if(!iw||!ih||ow>2*iw||oh>2*ih||out==lo||out==skip)throw std::runtime_error("upsample geometry/alias");LAUNCH(k_up_f32,(ow*oh*c+255)/256,256,lo,skip,scale,out,iw,ow,oh,c);
}
void launch_post_merge_f32(const float* lo,const float* pre,const float* scales,float* out,uint w,uint h,bool main8_low){LAUNCH(k_post_merge_f32,(w*h*32+255)/256,256,lo,pre,scales,out,w,h,main8_low);}
void launch_head_f32(const float* in,const float* w,const float* color,float* out,uint n){LAUNCH(k_head_f32,(n+63)/64,64,in,w,color,out,n);}
void launch_trace_f32(const float* in,uint n,float* stats,uint* bits){LAUNCH(k_trace_f32,1,256,in,n,stats,bits);}

} // namespace dlss5
