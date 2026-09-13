#pragma once
#include <hip/hip_runtime.h>
namespace dlss5 {
// Parameters are explicit color-space contracts, not pixel-magnitude guesses.
struct FrameSettings {
    float paper_white=1.f, transfer=1.f, color=1.f;
    bool display_srgb=false;
};
void frame_encode(const float* original_rgba,float* proxy_rgba,unsigned w,unsigned h,
                  FrameSettings settings,hipStream_t stream);
void frame_decode(const float* original_rgba,const float* proxy_rgba,const float* neural_rgb,
                  float* output_rgb,unsigned w,unsigned h,FrameSettings settings,hipStream_t stream);
// Full motion rectangle, caller-supplied displacement-to-UV scale. Reset or
// missing history/motion reproduces the network's current-color convention.
void frame_warp(const float* current_rgba,const float* history_rgb,const float* motion_xy,
                float* warped_rgba,unsigned w,unsigned h,unsigned ow,unsigned oh,
                unsigned mw,unsigned mh,float uv_scale_x,float uv_scale_y,bool reset,hipStream_t stream);
void frame_guard(const float* base_rgba,float* warped_rgba,unsigned pixels,float dark,float bright,hipStream_t stream);
void frame_smooth(const float* warped_rgba,float* rgb,unsigned pixels,float threshold,float strength,hipStream_t stream);
} // namespace dlss5
