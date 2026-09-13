// Differential and performance gate against the original scalar QKV kernel.
#include "dlss5_runtime.hpp"
#include "dlss5_launch.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
using namespace dlss5;
__global__ void reference_qkv(const float* input,const float* scales,u8* output,
 uint tokens,uint c,float qgain,bool half_squares){
 uint item=blockIdx.x*blockDim.x+threadIdx.x;
 if(item>=tokens*(c/32)*3)return;
 uint head=item%(c/32),part=(item/(c/32))%3;
 uint tok=item/(3*(c/32)),base=tok*3*c+part*c+head*32;
 float sum=0.f;
 if(part<2)for(uint ch=0;ch<32;++ch){float v=input[base+ch];volatile float square=half_squares?H(v*v):v*v;sum+=square;}
 float inv=part==2?1.f:rsqrtf(fmaxf(sum,6.198883056640625e-5f))*(part==0?scales[head]*qgain:1.f);
 for(uint ch=0;ch<32;++ch)output[base+ch]=e4m3_byte(input[base+ch]*inv);
}
int main(int argc,char**argv){try{
 int device=argc>1?std::atoi(argv[1]):0;hip_check(hipSetDevice(device),"device");
 hipDeviceProp_t p{};hip_check(hipGetDeviceProperties(&p,device),"props");printf("device=%s\n",p.name);
 bool perf=argc>2;float ratio=0;
 for(uint c:{32u,64u,128u,256u,512u,1024u})for(uint tokens:{1u,65u,32768u})for(bool half:{false,true}){
 size_t n=size_t(tokens)*3*c;std::vector<float> in(n),s(c/32);unsigned z=0x12345678;
 for(size_t i=0;i<n;++i){z=z*1664525u+1013904223u;in[i]=(int(z>>16)-32768)*.000125f;if(i%97==0)in[i]=0;}
 for(uint i=0;i<c/32;++i)s[i]=.5f+i*.03125f;
 DeviceBuf di(n*4),ds(s.size()*4),a(n+64),b(n+64);di.upload(in.data(),n*4);ds.upload(s.data(),s.size()*4);
 hip_check(hipMemset(a.p,0xa5,n+64),"sentinel");hip_check(hipMemset(b.p,0xa5,n+64),"sentinel");
 uint count=tokens*(c/32)*3;
 auto ref=[&]{reference_qkv<<<(count+63)/64,64>>>(di.as<float>(),ds.as<float>(),a.as<u8>(),tokens,c,5.65625f,half);};
 auto opt=[&]{launch_normalize_qkv(di.as<float>(),ds.as<float>(),b.as<u8>(),tokens,c,5.65625f,half);};
 ref();opt();hip_check(hipDeviceSynchronize(),"run");std::vector<u8>x(n+64),y(n+64);a.download(x.data(),x.size());b.download(y.data(),y.size());
 if(x!=y)throw std::runtime_error("QKV byte mismatch or overrun");
 if(tokens==32768&&c==32&&half){hipEvent_t start,end;hip_check(hipEventCreate(&start),"event");hip_check(hipEventCreate(&end),"event");
 auto time=[&](auto fn){hip_check(hipEventRecord(start),"start");for(int r=0;r<20;++r)fn();hip_check(hipEventRecord(end),"end");hip_check(hipEventSynchronize(end),"wait");float ms;hip_check(hipEventElapsedTime(&ms,start,end),"elapsed");return ms/20;};
 float old=time(ref),now=time(opt);ratio=old/now;printf("QKV reference_ms=%.6f candidate_ms=%.6f speedup=%.3f\n",old,now,ratio);hip_check(hipEventDestroy(start),"destroy");hip_check(hipEventDestroy(end),"destroy");}
 }
 printf("PASS QKV 36 cases exact bytes and guard\n");
 if(perf&&ratio<1.3f)throw std::runtime_error("QKV performance regression: need >=1.3x scalar baseline");
 return 0;
 }catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
