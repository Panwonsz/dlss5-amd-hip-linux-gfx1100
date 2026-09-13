// Explicitly SYNTHETIC tables, never a substitute for mapped NVIDIA weights.
#include "dlss5_network.hpp"
#include "dlss5_launch.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <numeric>
using namespace dlss5;
namespace fs=std::filesystem;
static void table(const fs::path& d,const std::string& name,size_t n,size_t scale=0){
 std::vector<float> v(n,0);if(scale)std::fill(v.end()-scale,v.end(),1.f);
 if(name=="block0-ffn") {v[0*16+4]=.0625f;v[1*16+13]=1;std::fill(v.end()-32,v.end(),.5f);}
 if(name=="post70-scales") {std::fill(v.begin(),v.begin()+32,0.f);std::fill(v.begin()+32,v.end(),.75f);}
 if(name=="post70-head") {v[0]=.5f;v[32+1]=.5f;}
 // Nonzero signal reaches EVERY stage despite zero attention/FFN branches.
 if(name.find("-ds")!=std::string::npos||name=="head-matrix") {size_t cin=size_t(std::sqrt(double(n)/2));for(size_t j=0;j<cin*2;++j)v[j*cin+j%cin]=1;}
 if(name.find("-weights")!=std::string::npos) {size_t cout=scale,cin=(n-cout)/cout;for(size_t j=0;j<cout;++j)v[j*cin+j%cin]=1;}
 std::ofstream f(d/(name+".f32"),std::ios::binary);f.write((char*)v.data(),v.size()*4);
}
static void mh(const fs::path&d,int b,int c){std::string p=b==70?"post70":"block"+std::to_string(b);
 table(d,p+"-ffn",c==32?8736:9ull*c*c+c,c);
 table(d,p+"-attention",4ull*c*c+(c/32)*4096+c/32+c,c);
}
static void fixture(const fs::path&d){fs::create_directories(d);
 for(int b=0;b<=70;++b){if(b<=4||b>=66)mh(d,b,32);else if(b<=8||b>=62)mh(d,b,64);else if(b<=14||b>=56)mh(d,b,128);else if(b<=22||(b>=48&&b<=55))mh(d,b,256);else if((b>=23&&b<=30)||(b>=40&&b<=47)){auto p="block"+std::to_string(b);table(d,p+"-ffwd",524288);table(d,p+"-ffwd-projection",262656,512);table(d,p+"-attention",1114640,512);}else if(b>=31&&b<=38){auto p="block"+std::to_string(b);table(d,p+"-expand",4194304);table(d,p+"-contract",4195328,1024);table(d,p+"-qkv",3145760,32);table(d,p+"-projection",1049600,1024);}}
 for(auto bc: {std::pair<int,int>{4,32},{8,64},{14,128},{22,256}})table(d,"block"+std::to_string(bc.first)+"-ds",2ull*bc.second*bc.second);
 table(d,"head-matrix",1024*512);table(d,"decoder39-weights",1024*512+512,512);
 for(auto bc:{std::pair<int,int>{48,256},{56,128},{62,64},{66,32}})table(d,"block"+std::to_string(bc.first)+"-weights",2ull*bc.second*bc.second+bc.second,bc.second);
 table(d,"post70-scales",64);table(d,"post70-head",96);
 std::vector<int> map(655360);std::iota(map.begin(),map.end(),0);
 for(auto name:{"hwc-to-vit.i32","vit-to-hwc.i32"}){std::ofstream f(d/name,std::ios::binary);f.write((char*)map.data(),map.size()*4);}
}
int main(int argc,char**argv){try{
 fs::path d=argc>1?argv[1]:std::string(std::getenv("DLSS5_GRAPH_TEST_BUILD")?std::getenv("DLSS5_GRAPH_TEST_BUILD"):"linux/build/graph-audit")+"/synthetic";fixture(d);
 std::vector<float> in(1920ull*1080*4),out(1920ull*1152*3),again(out.size());
 for(size_t p=0;p<in.size()/4;++p){in[p*4]=float(p%53)/64.f;in[p*4+1]=.25f;in[p*4+2]=.75f;in[p*4+3]=1;}
 setenv("DLSS5_HIP_TRACE","1",1);
 hipDeviceProp_t prop{};int device=0;hip_check(hipGetDevice(&device),"device");hip_check(hipGetDeviceProperties(&prop,device),"properties");size_t free0,total,free1;hip_check(hipMemGetInfo(&free0,&total),"memory before");
 Network70 net;net.create(d.string());hip_check(hipMemGetInfo(&free1,&total),"memory after");std::printf("device=%d %s allocated_delta_bytes=%zu total_bytes=%zu\n",device,prop.name,free0-free1,total);net.run(in.data(),out.data(),7);
 auto counts=net.last_block_counts();auto trace=net.last_trace();
 for(int b=0;b<71;++b)if(counts[b]!=1||!trace[b].checked||trace[b].nonfinite||!trace[b].elements||trace[b].minimum==trace[b].maximum)throw std::runtime_error("all71 finite/nonconstant trace: "+std::to_string(b));
 size_t bad=0;for(unsigned y=0;y<1152;++y)for(unsigned x=0;x<1920;++x){unsigned iy=y<1080?y:2158-y;size_t p=size_t(y)*1920+x;
 float prefix=H(H(H(in[(size_t(iy)*1920+x)*4])-.5f)*.125f);
 float pre=from_e4m3(e4m3_byte(H(prefix*.5f)));
 float body=H(pre*.75f),acc=H(body*.5f);
 float expected=std::clamp((acc*.03125f+(.25f*.125f-.0625f))*8.f+.5f,0.f,1.f);
 for(int c=0;c<3;++c)if(!std::isfinite(out[p*3+c]))++bad;
 if(out[p*3+1]!=expected){if(bad<4)std::printf("block0/skip/head mismatch p=%zu got=%g expected=%g\n",p,out[p*3+1],expected);++bad;}}
 std::printf("synthetic 1080 network mismatches=%zu GPU_ms=%.3f\n",bad,net.last_gpu_ms());if(bad)return 1;
 net.run(in.data(),again.data(),19);size_t changed=0;for(size_t i=0;i<out.size();++i)changed+=out[i]!=again[i];
 std::printf("seed changes=%zu GPU_ms=%.3f\n",changed,net.last_gpu_ms());if(!changed)return 1;
 // Same seed deterministic on a normal unsynchronized graph stream.
 unsetenv("DLSS5_HIP_TRACE");net.run(in.data(),again.data(),7);if(out!=again)throw std::runtime_error("same seed changed");
 // Explicit synthetic prefix channel 13 consumes temporal R (not a no-op setter).
 std::vector<float> history(1920ull*1152*4,.875f),temporal(out.size());
 auto run_history=[&](float* dst){
#ifdef DLSS5_CONFIGURED_POLICIES
 net.run(in.data(),dst,7,history.data());
#else
 net.run(in.data(),dst,7);
#endif
 };
 run_history(temporal.data());if(temporal==out)throw std::runtime_error("temporal RGB ignored");
 float hp=H(H(H(.875f)-.5f)*.125f),hf=from_e4m3(e4m3_byte(H(hp*.5f)));
 float hwant=std::clamp((H(H(hf*.75f)*.5f)*.03125f+(.25f*.125f-.0625f))*8.f+.5f,0.f,1.f);
 for(size_t p=0;p<temporal.size()/3;++p)if(temporal[p*3+1]!=hwant)throw std::runtime_error("synthetic temporal oracle mismatch");
 run_history(again.data());if(temporal!=again)throw std::runtime_error("same history changed");
 net.run(in.data(),again.data(),7);if(out!=again)throw std::runtime_error("history leaked into reset/no-history");
 std::printf("synthetic temporal nonzero/oracle/same-history/reset-nohistory PASS\n");
 // Exact size, finite coefficients, maps and failure-safe repeated create.
 auto reject=[&](const char*label){bool threw=false;try{net.create(d.string());}catch(const std::exception&){threw=true;}if(!threw)throw std::runtime_error(std::string("accepted corrupt ")+label);};
 auto scale_path=d/"post70-scales.f32";
 {std::ofstream f(scale_path,std::ios::binary|std::ios::app);float z=0;f.write((char*)&z,4);}reject("oversize");table(d,"post70-scales",64);
 {std::fstream f(scale_path,std::ios::binary|std::ios::in|std::ios::out);float z=std::numeric_limits<float>::quiet_NaN();f.write((char*)&z,4);}reject("NaN");table(d,"post70-scales",64);
 {std::fstream f(d/"hwc-to-vit.i32",std::ios::binary|std::ios::in|std::ios::out);int z=-1;f.write((char*)&z,4);}reject("negative map");
 {std::fstream f(d/"hwc-to-vit.i32",std::ios::binary|std::ios::in|std::ios::out);int z=1;f.write((char*)&z,4);}reject("duplicate map");
 {std::fstream f(d/"hwc-to-vit.i32",std::ios::binary|std::ios::in|std::ios::out);int z[2]={1,0};f.write((char*)z,8);}reject("inverse map");
 {std::fstream f(d/"hwc-to-vit.i32",std::ios::binary|std::ios::in|std::ios::out);int z[2]={0,1};f.write((char*)z,8);}
 // Existing bad f32 must not fall back to good f16.
 {std::vector<uint16_t> v(64);for(int i=32;i<64;++i)v[i]=0x3a00;std::ofstream f(d/"post70-scales.f16",std::ios::binary);f.write((char*)v.data(),128);}
 {std::ofstream f(scale_path,std::ios::binary);char z=0;f.write(&z,1);}reject("f32 fallback");fs::remove(scale_path);
 net.create(d.string());net.run(in.data(),again.data(),7);if(out!=again)throw std::runtime_error("f16/recreate changed output");
 std::printf("all71 count/finite/nonconstant PASS; same-seed PASS; loader/recreate/f16 PASS; normal GPU_ms=%.3f\n",net.last_gpu_ms());
 table(d,"post70-scales",64);
 return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"network regression FAIL: %s\n",e.what());return 1;}}
