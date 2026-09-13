#include "dlss5_runtime.hpp"
#include <cstdio>
#include <vector>
using namespace dlss5;
__host__ __device__ inline u8 reference_fp8(float v) {
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


__global__ void convert(const float* in,u8* out,uint n,bool reference){uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=reference?reference_fp8(in[i]):e4m3_byte(in[i]);}
int main(int argc,char**argv){try{
 hip_check(hipSetDevice(argc>1?std::atoi(argv[1]):0),"device");
 std::vector<float>x;unsigned z=123;
 for(unsigned i=0;i<4000000;++i){z=z*1664525u+1013904223u;x.push_back(as_f32(z));}
 for(unsigned h=0;h<65536;++h)x.push_back(half_to_float(h));
 for(unsigned a=0x39000000;a<0x44000000;a+=0x80000)for(int d=-2;d<=2;++d){x.push_back(as_f32(a+d));x.push_back(as_f32((a+d)|0x80000000u));}
 DeviceBuf in(x.size()*4),a(x.size()),b(x.size());in.upload(x.data(),x.size()*4);
 auto ref=[&]{convert<<<(x.size()+255)/256,256>>>(in.as<float>(),a.as<u8>(),x.size(),true);};
 auto opt=[&]{convert<<<(x.size()+255)/256,256>>>(in.as<float>(),b.as<u8>(),x.size(),false);};
 ref();opt();hip_check(hipDeviceSynchronize(),"run");std::vector<u8> aa(x.size()),bb(x.size());a.download(aa.data(),aa.size());b.download(bb.data(),bb.size());
 for(size_t i=0;i<x.size();++i)if(aa[i]!=bb[i]){printf("bits=%08x expected=%02x actual=%02x\n",as_u32(x[i]),aa[i],bb[i]);throw std::runtime_error("FP8 mismatch");}
 hipEvent_t start,end;hip_check(hipEventCreate(&start),"event");hip_check(hipEventCreate(&end),"event");
 auto time=[&](auto fn){hip_check(hipEventRecord(start),"start");for(int i=0;i<40;++i)fn();hip_check(hipEventRecord(end),"end");hip_check(hipEventSynchronize(end),"sync");float ms;hip_check(hipEventElapsedTime(&ms,start,end),"ms");return ms/40;};
 float old=time(ref),now=time(opt);printf("PASS FP8 %zu values exact reference_ms=%.6f candidate_ms=%.6f speedup=%.3f\n",x.size(),old,now,old/now);
 hip_check(hipEventDestroy(start),"destroy");hip_check(hipEventDestroy(end),"destroy");
 if(argc>2&&old/now<1.1f)throw std::runtime_error("FP8 performance: need >=1.1x baseline");return 0;
 }catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
