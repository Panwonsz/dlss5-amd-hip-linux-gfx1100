#include "dlss5_capi.h"
#include <cstdio>
#include <cmath>
#include <limits>
#include <vector>
int main(int argc,char**argv){
    if(argc!=2)return 2;
    Dlss5Frame frame{};frame.struct_size=sizeof frame;
    if(dlss5_run_frame(&frame)!=-1)return 1;
    if(dlss5_init(argv[1],0)){std::fprintf(stderr,"%s\n",dlss5_last_error());return 1;}
    const size_t n=1920ull*1080;
    std::vector<float> rgba(n*4),rgb(n*3),first(n*3),motion(64*36*2,0.f);
    for(size_t i=0;i<n;++i){rgba[i*4]=.2f+float(i%37)*.003f;rgba[i*4+1]=.15f;rgba[i*4+2]=.1f;rgba[i*4+3]=.75f;}
    frame.flags=1;frame.rgba=rgba.data();frame.rgb=rgb.data();frame.motion=motion.data();frame.motion_width=64;frame.motion_height=36;
    frame.motion_uv_scale_x=1.f/1920;frame.motion_uv_scale_y=1.f/1080;frame.paper_white=frame.transfer=frame.color=1;
    frame.seed=2;frame.reset=1;
    if(dlss5_run_frame(&frame)){std::fprintf(stderr,"first %s\n",dlss5_last_error());return 1;}first=rgb;
    frame.reset=0;
    if(dlss5_run_frame(&frame)){std::fprintf(stderr,"history %s\n",dlss5_last_error());return 1;}
    size_t changed=0;for(size_t i=0;i<rgb.size();++i){if(!std::isfinite(rgb[i]))return 1;changed+=rgb[i]!=first[i];}
    if(!changed){std::fprintf(stderr,"history had no effect\n");return 1;}
    frame.reset=1;if(dlss5_run_frame(&frame)||rgb!=first)return 1;
    // Every rejected frame breaks temporal continuity, including early ABI
    // returns and calls through the legacy no-history API.
    for(int rejection=0;rejection<4;++rejection){
        frame.reset=1;frame.flags=1;
        if(dlss5_run_frame(&frame)||rgb!=first)return 1;
        rgb.assign(rgb.size(),-7.f);
        if(rejection==0){
            frame.flags=2;
            if(dlss5_run_frame(&frame)!=-1)return 1;
            frame.flags=1;
        }else if(rejection==1){
            if(dlss5_run_frame(nullptr)!=-1)return 1;
        }else if(rejection==2){
            float saved=rgba[0];rgba[0]=std::numeric_limits<float>::quiet_NaN();
            if(dlss5_run(rgba.data(),rgb.data(),2)!=-1)return 1;
            rgba[0]=saved;
        }else if(dlss5_run(nullptr,rgb.data(),2)!=-1)return 1;
        for(float v:rgb)if(v!=-7.f)return 1;
        frame.reset=0;
        if(dlss5_run_frame(&frame)||rgb!=first){
            std::fprintf(stderr,"stale history after rejection=%d\n",rejection);return 1;
        }
    }
    rgb.assign(rgb.size(),-7.f);rgba[0]=std::numeric_limits<float>::quiet_NaN();
    if(dlss5_run_frame(&frame)!=-1)return 1;
    for(float v:rgb)if(v!=-7.f)return 1;
    dlss5_shutdown();
    std::printf("frame C API reset/history/finite/atomic rejection PASS changed=%zu\n",changed);
    return 0;
}
