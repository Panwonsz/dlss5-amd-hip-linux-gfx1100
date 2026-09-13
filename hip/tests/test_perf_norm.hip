// Original scalar oracle and event-timed regression; device 1, offline only.
#include "dlss5_runtime.hpp"
#include "dlss5_launch.hpp"
#include <sqlite3.h>
#include <random>
#include <cstdio>
#include <cstring>
using namespace dlss5;
__global__ void old_norm(const float* input,const float* scales,u8* output,uint tokens,uint c,float qgain,bool half_squares){
 uint item=blockIdx.x*blockDim.x+threadIdx.x;
 if(item>=tokens*(c/32)*3)return;
 uint head=item%(c/32),part=(item/(c/32))%3;
 uint tok=item/(3*(c/32)),base=tok*3*c+part*c+head*32;
 float sum=0.f;
 if(part<2)for(uint ch=0;ch<32;++ch){float v=input[base+ch];volatile float square=half_squares?H(v*v):v*v;sum+=square;}
 float inv=part==2?1.f:rsqrtf(fmaxf(sum,6.198883056640625e-5f))*(part==0?scales[head]*qgain:1.f);
 for(uint ch=0;ch<32;++ch)output[base+ch]=e4m3_byte(input[base+ch]*inv);
}
template<class F> float timed(F fn){hipEvent_t a,b;hip_check(hipEventCreate(&a),"event");hip_check(hipEventCreate(&b),"event");fn();hip_check(hipDeviceSynchronize(),"warmup");hip_check(hipEventRecord(a),"record");for(int i=0;i<12;++i)fn();hip_check(hipEventRecord(b),"record");hip_check(hipEventSynchronize(b),"sync");float ms;hip_check(hipEventElapsedTime(&ms,a,b),"elapsed");hipEventDestroy(a);hipEventDestroy(b);return ms/12;}
int main(){try{
 sqlite3* db=nullptr;
 if(sqlite3_open_v2("linux/build/live/profile/network_results.db",&db,SQLITE_OPEN_READONLY,nullptr)!=SQLITE_OK)return 2;
 auto cb=[](void*,int n,char** v,char** col)->int{for(int i=0;i<n;++i)printf("%s=%s ",col[i],v[i]?v[i]:"NULL");puts("");return 0;};
 char* err=nullptr;
 for(const char* q:{"SELECT name,sql FROM sqlite_master WHERE name IN ('top_kernels','kernels')","SELECT * FROM top_kernels LIMIT 12","SELECT * FROM kernels LIMIT 1"})if(sqlite3_exec(db,q,cb,nullptr,&err)!=SQLITE_OK){fprintf(stderr,"sqlite: %s\n",err);sqlite3_free(err);return 2;}
 sqlite3_close(db);
 hip_check(hipSetDevice(1),"device1");unsigned badgroups=0;bool perf=true;
 for(uint c:{32u,64u,128u,256u,512u,1024u}){
 uint tokens=c==32?65537:2051;size_t count=size_t(tokens)*3*c;
 std::vector<float> in(count),sc(c/32);std::mt19937 rng(419);
 for(float&v:in)v=std::ldexp(float(int(rng()%65537)-32768)/32768.f,int(rng()%22)-10);
 for(float&v:sc)v=float(int(rng()%31)-15)/7.f;
 const unsigned edges[]={0,0x80000000,0x7f800000,0xff800000,0x7fc00000,0xffc12345,1,0x80000001};
 for(unsigned i=0;i<sizeof(edges)/sizeof(*edges);++i)in[i*32]=as_f32(edges[i]);
 DeviceBuf di(count*4),ds(sc.size()*4),a(count),b(count);di.upload(in.data(),count*4);ds.upload(sc.data(),sc.size()*4);
 for(bool half:{false,true}){
 auto ref=[&]{old_norm<<<(tokens*(c/32)*3+63)/64,64>>>(di.as<float>(),ds.as<float>(),a.as<u8>(),tokens,c,.73f,half);};
 auto got=[&]{launch_normalize_qkv(di.as<float>(),ds.as<float>(),b.as<u8>(),tokens,c,.73f,half);};
 float oldms=timed(ref),newms=timed(got);std::vector<u8>x(count),y(count);a.download(x.data(),count);b.download(y.data(),count);size_t bad=0;for(size_t i=0;i<count;++i)bad+=x[i]!=y[i];badgroups+=bad!=0;
 printf("norm C=%u tokens=%u half=%d old_ms=%.6f new_ms=%.6f ratio=%.4f mismatches=%zu\n",c,tokens,half,oldms,newms,newms/oldms,bad);
 if(c==32)perf&=newms<oldms*.75f;
 }
 }
 printf("exact_groups_failed=%u performance_threshold_0.75=%s\n",badgroups,perf?"PASS":"FAIL");return badgroups||!perf?1:0;
 }catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 2;}}
