#include "dlss5_launch.hpp"
#include "dlss5_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace dlss5;
static unsigned bad;
static void same(float a,float b,const char*name){if(!std::isfinite(a)||a!=b){if(bad<5)std::printf("%s actual=%g expected=%g\n",name,a,b);++bad;}}
int main(){
    // Distinct Q/K norms and unequal head scales; values must not normalize.
    const uint n=5,c=64;
    std::vector<float> q(n*3*c),sc{.3f,1.7f};std::vector<u8> packed(q.size());
    for(size_t i=0;i<q.size();++i)q[i]=float(int((i*7)%31)-15)*.071f;
    DeviceBuf dq(q.size()*4),ds(sc.size()*4),dp(packed.size());
    dq.upload(q.data(),q.size()*4);ds.upload(sc.data(),sc.size()*4);
    launch_normalize_qkv(dq.as<float>(),ds.as<float>(),dp.as<u8>(),n,c,1.f);
    hip_check(hipDeviceSynchronize(),"normalize");dp.download(packed.data(),packed.size());
    for(uint t=0;t<n;++t)for(uint part=0;part<3;++part)for(uint head=0;head<2;++head){
        float sum=0;for(uint ch=0;ch<32;++ch){float v=q[t*3*c+part*c+head*32+ch];sum+=H(v*v);}
        float gain=part==2?1.f:(part==0?sc[head]:1.f)/std::sqrt(std::max(sum,6.198883056640625e-5f));
        for(uint ch=0;ch<32;++ch){size_t i=t*3*c+part*c+head*32+ch;same(from_e4m3(packed[i]),from_e4m3(e4m3_byte(q[i]*gain)),"norm");}
    }
    // Shifted windows must zero border, round-trip, and preserve FP16 residuals.
    const uint w=16,h=8,channels=32,sw=24,sh=16;
    std::vector<float> a(w*h*channels),b(sw*sh*channels),back(a.size());
    for(size_t i=0;i<a.size();++i)a[i]=float(i+1)*.000731f;
    DeviceBuf da(a.size()*4),db(b.size()*4),dc(a.size()*4);
    da.upload(a.data(),a.size()*4);
    launch_reframe_f32(da.as<float>(),db.as<float>(),w,h,sw,sh,4,4,channels,0,0);
    launch_reframe_f32(db.as<float>(),dc.as<float>(),w,h,sw,sh,4,4,channels,1,0);
    hip_check(hipDeviceSynchronize(),"reframe");db.download(b.data(),b.size()*4);dc.download(back.data(),back.size()*4);
    for(size_t i=0;i<a.size();++i)same(back[i],a[i],"roundtrip");
    for(uint y=0;y<sh;++y)for(uint x=0;x<sw;++x)for(uint ch=0;ch<channels;++ch){
        uint win=(y/8)*(sw/8)+x/8,tok=(y%8)*8+x%8;
        float e=x>=4&&y>=4&&x<w+4&&y<h+4?a[((y-4)*w+x-4)*channels+ch]:0.f;
        same(b[(win*64+tok)*channels+ch],e,"border");
    }
    // Pool the raw residual, not a prematurely quantized FP8 stream.
    std::vector<float> pooled(w/2*h/2*channels);
    launch_pool_f32(da.as<float>(),dc.as<float>(),w,h,channels,w/2,h/2);
    hip_check(hipDeviceSynchronize(),"pool");dc.download(pooled.data(),pooled.size()*4);
    for(uint y=0;y<h/2;++y)for(uint x=0;x<w/2;++x)for(uint ch=0;ch<channels;++ch){
        uint i=(y*2*w+x*2)*channels+ch;
        float e=H(H(H(a[i]+a[i+channels])+H(a[i+w*channels]+a[i+(w+1)*channels]))*.25f);
        same(pooled[(y*w/2+x)*channels+ch],from_e4m3(e4m3_byte(e)),"raw pool");
    }
    std::printf("logical norm/reframe/pool mismatches=%u\n",bad);return bad?1:0;
}
