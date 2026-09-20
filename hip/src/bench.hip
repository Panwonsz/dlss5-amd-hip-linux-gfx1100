#include "dlss5_runtime.hpp"
#include "dlss5_launch.hpp"
#include "dlss5_network.hpp"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

using namespace dlss5;
namespace {
double now_ms() {
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
uint number(const std::string& text,uint max,const char* name) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error(std::string("invalid ") + name);
    auto n = std::stoull(text);
    if (n > max) throw std::runtime_error(std::string("out of range ") + name);
    return uint(n);
}
}
int main(int argc,char** argv) {
 try {
    std::string weights,input,output;
    uint seed=0,runs=3,device=0;
    bool trace=false;
    for (int i=1;i<argc;++i) {
        std::string a=argv[i];
        if (a=="--help") {
            std::puts("usage: hip-network70 [--weights DIR] [--input RGBA32F] [--output RGB32F]\n"
                      "  --input: 1920x1080 little-endian interleaved float RGBA in network color space\n"
                      "  --output: cropped 1920x1080 little-endian float RGB; never replaces input\n"
                      "  --runs N (1..1000), --seed UINT32, --device N, --trace (all 71 blocks)\n"
                      "  With no weights: run only the explicitly synthetic FFN microbenchmark.\n"
                      "  With weights but no input: use an explicitly synthetic RGB gradient.");
            return 0;
        }
        if (a=="--trace") {trace=true;continue;}
        if (a!="--weights" && a!="--input" && a!="--output" && a!="--runs" && a!="--seed" && a!="--device")
            throw std::runtime_error("unknown option: "+a);
        if (++i==argc) throw std::runtime_error("missing value: "+a);
        std::string v=argv[i];
        if(a=="--weights")weights=v;
        else if(a=="--input")input=v;
        else if(a=="--output")output=v;
        else if(a=="--runs") {runs=number(v,1000,"runs");if(!runs)throw std::runtime_error("runs must be positive");}
        else if(a=="--seed")seed=number(v,std::numeric_limits<uint>::max(),"seed");
        else device=number(v,255,"device");
    }
    if(weights.empty()&&(!input.empty()||!output.empty()||trace))throw std::runtime_error("--weights is required for input/output/trace");
    if(!output.empty()&&std::filesystem::exists(output))throw std::runtime_error("output exists; choose a new path");
    std::vector<float> rgb;
    if(!weights.empty()) {
        if(input.empty()) {
            rgb.resize(1920ull*1080*4);
            for(size_t p=0;p<1920ull*1080;++p) {
                rgb[p*4]=float(p%1920)/1919.f;rgb[p*4+1]=float(p/1920)/1079.f;
                rgb[p*4+2]=.25f;rgb[p*4+3]=1.f;
            }
        } else {
            rgb=read_f32(input);
            if(rgb.size()!=1920ull*1080*4)throw std::runtime_error("input must be 1920x1080 RGBA32F");
            for(float v:rgb)if(!std::isfinite(v))throw std::runtime_error("nonfinite input");
        }
    }
    hip_check(hipSetDevice(int(device)),"setDevice");hipDeviceProp_t prop{};
    hip_check(hipGetDeviceProperties(&prop,int(device)),"props");
    // weights= is on this line because a timing without its configuration is not a timing. An
    // evening went into suspecting a regression that was an unset DLSS5_W16: the same binary reads
    // 186 ms on E4M3 weights and 132 on f16, and nothing in the output said which had run.
    std::printf("HIP DLSS5 device=%s arch=%s vram=%.2f GiB weights=%s kernels[%s]\n",prop.name,
                prop.gcnArchName,double(prop.totalGlobalMem)/(1024*1024*1024),
                dlss5_w16()?"f16":"e4m3",dlss5_kernel_formats());
    if(weights.empty()) {
        const int c=64;const uint n=4096;
        std::vector<float> fw(8ull*c*c);for(size_t i=0;i<fw.size();++i)fw[i]=float(int(i%17)-8)*.015625f;
        std::vector<u8> wt(fw.size()),in(n*c,e4m3_byte(.5f));
        pack_tiled_e4m3(wt.data(),fw.data(),4*c,c);pack_tiled_e4m3(wt.data()+4*c*c,fw.data()+4*c*c,c,4*c);
        DeviceBuf di(in.size()),dw(wt.size()),out(in.size());di.upload(in.data(),in.size());dw.upload(wt.data(),wt.size());
        launch_ffn(c,di.as<u8>(),dw.as<u8>(),out.as<u8>(),n);hip_check(hipDeviceSynchronize(),"warmup");
        double t=now_ms();for(int r=0;r<20;++r)launch_ffn(c,di.as<u8>(),dw.as<u8>(),out.as<u8>(),n);
        hip_check(hipDeviceSynchronize(),"microbench");
        std::printf("synthetic FFN C64 tokens=4096 host_ms_per_run=%.4f\n",(now_ms()-t)/20);
        return 0;
    }
    if(trace)setenv("DLSS5_HIP_TRACE","1",1);
    Network70 net;double begin=now_ms();net.create(weights);
    std::printf("model_init_ms=%.2f input=%s\n",now_ms()-begin,input.empty()?"synthetic-gradient":input.c_str());
    std::vector<float> out(1920ull*1152*3),first;
    for(uint r=0;r<runs;++r) {
        double t=now_ms();net.run(rgb.data(),out.data(),seed);double host=now_ms()-t;
        auto counts=net.last_block_counts();for(uint b=0;b<71;++b)if(counts[b]!=1)throw std::runtime_error("missing/repeated block "+std::to_string(b));
        double sum=0,delta=0;float mn=INFINITY,mx=-INFINITY;size_t changed=0;
        for(size_t p=0;p<1920ull*1080;++p)for(uint c=0;c<3;++c){float v=out[p*3+c];if(!std::isfinite(v))throw std::runtime_error("nonfinite RGB output");
            mn=std::min(mn,v);mx=std::max(mx,v);sum+=v;delta+=std::fabs(double(v)-rgb[p*4+c]);changed+=v!=rgb[p*4+c];}
        bool replay=r==0||out==first;if(!replay)throw std::runtime_error("same input/seed produced different output");if(r==0)first=out;
        std::printf("network71 run=%u seed=%u host_ms=%.3f gpu_ms=%.3f blocks=71 finite=1 replay=1 min=%g max=%g mean=%.7f mean_abs_change=%.7f changed=%zu\n",
                    r,seed,host,net.last_gpu_ms(),mn,mx,sum/(1920*1080*3),delta/(1920*1080*3),changed);
    }
    if(!output.empty()) {
        std::ofstream f(output,std::ios::binary);f.write((const char*)out.data(),1920ull*1080*3*4);f.close();
        if(!f)throw std::runtime_error("output write failed: "+output);
        std::printf("output=%s bytes=%zu\n",output.c_str(),size_t(1920)*1080*3*4);
    }
    return 0;
 } catch(const std::exception&e) {std::fprintf(stderr,"hip-network70: %s\n",e.what());return 1;}
}