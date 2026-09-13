#include "dlss5_frame.hpp"
#include "dlss5_runtime.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace dlss5;
using V=std::array<double,3>;
static unsigned failures;
static V mul(const double* m,V x){V o{};for(int r=0;r<3;r++)for(int c=0;c<3;c++)o[r]+=m[r*3+c]*x[c];return o;}
static V lab(V c){const double a[]={.4122214708,.5363325363,.0514459929,.2119034982,.6806995451,.1073969566,.0883024619,.2817188376,.6299787005};const double b[]={.2104542553,.7936177850,-.0040720468,1.9779984951,-2.4285922050,.4505937099,.0259040371,.7827717662,-.8086757660};auto v=mul(a,c);for(double&x:v)x=std::cbrt(x);return mul(b,v);}
static V unlab(V c){const double a[]={1,.3963377774,.2158037573,1,-.1055613458,-.0638541728,1,-.0894841775,-1.2914855480};const double b[]={4.0767416621,-3.3077115913,.2309699292,-1.2684380046,2.6097574011,-.3413193965,-.0041960863,-.7034186147,1.7076147010};auto v=mul(a,c);for(double&x:v)x=x*x*x;return mul(b,v);}
static double lum(V c){return c[0]*.212639+c[1]*.715169+c[2]*.072192;}
static double linear(double c){c=std::clamp(c,0.,1.);return c<=.04045?c/12.92:std::pow((c+.055)/1.055,2.4);}
static double srgb(double c){return c<=.0031308?c*12.92:1.055*std::pow(c,1./2.4)-.055;}
static V decode(V original,V proxy,V neural,const FrameSettings&s){
    for(int c=0;c<3;c++){original[c]=s.display_srgb?linear(original[c]):std::max(original[c],0.)/s.paper_white;proxy[c]=linear(proxy[c]);neural[c]=linear(neural[c]);}
    double oy=lum(original),py=lum(proxy),ny=lum(neural);V result=original;
    if(ny>1e-5){double ratio=oy<py?oy/std::max(py,1e-6):(ny+std::max(0.,oy-py))/ny;V incorrect=neural;for(double&x:incorrect)x*=ratio;
        V a=lab(incorrect),b=lab(neural);double ca=std::hypot(a[1],a[2]),cb=std::hypot(b[1],b[2]);a[1]=b[1]*(cb==0?1:ca/cb);a[2]=b[2]*(cb==0?1:ca/cb);V rgb=unlab(a);
        const double m1[]={.613097,.339523,.047379,.070194,.916354,.013452,.020616,.109570,.869815},m2[]={1.705051,-.621792,-.083259,-.130256,1.140805,-.010548,-.024003,-.128969,1.152972};rgb=mul(m1,rgb);for(double&x:rgb)x=std::max(0.,x);rgb=mul(m2,rgb);
        for(int c=0;c<3;c++)result[c]=original[c]+s.transfer*(rgb[c]-original[c]);}
    double ratio=oy==0?1:std::clamp(lum(result)/oy,0.,4.);
    for(int c=0;c<3;c++){result[c]=original[c]*ratio+s.color*(result[c]-original[c]*ratio);result[c]=s.display_srgb?srgb(std::clamp(result[c],0.,1.)):result[c]*s.paper_white;}
    return result;
}
static void close(float got,double expected,const char*where,double tolerance=3e-5){if(!std::isfinite(got)||std::abs(got-expected)>tolerance*(1+std::abs(expected))){if(failures<8)std::printf("FAIL %s got=%g expected=%.9g\n",where,got,expected);++failures;}}
int main(){
    const unsigned w=8,h=4,n=w*h;
    std::vector<float> original(n*4),proxy(n*4),neural(n*3),out(n*3);
    for(unsigned i=0;i<n;i++){for(unsigned c=0;c<3;c++){original[i*4+c]=float(int((i*7+c*11)%37)-3)*.09f;neural[i*3+c]=float((i*13+c*7)%29)/28.f;}original[i*4+3]=float(i%5)*.2f;}
    DeviceBuf di(n*16),dp(n*16),dn(n*12),dout(n*12);di.upload(original.data(),n*16);dn.upload(neural.data(),n*12);
    for(bool display:{false,true})for(float white:{.5f,1.f,2.f}){
        FrameSettings s;s.display_srgb=display;s.paper_white=white;s.transfer=.8f;s.color=.7f;
        frame_encode(di.as<float>(),dp.as<float>(),w,h,s,nullptr);
        frame_decode(di.as<float>(),dp.as<float>(),dn.as<float>(),dout.as<float>(),w,h,s,nullptr);
        hip_check(hipDeviceSynchronize(),"codec");dp.download(proxy.data(),n*16);dout.download(out.data(),n*12);
        for(unsigned i=0;i<n;i++){
            for(unsigned c=0;c<3;c++){double v=std::max(double(original[i*4+c]),0.)/white;
                if(v>.75)v=1-.25*std::exp(-5.770780*(v-.75));v=srgb(std::clamp(v,0.,1.));
                if(display)v=std::clamp(double(original[i*4+c]),0.,1.);close(proxy[i*4+c],H(float(v)),"encode",1e-6);}
            close(proxy[i*4+3],1,"proxy-alpha");V a,b,c;for(int j=0;j<3;j++){a[j]=original[i*4+j];b[j]=proxy[i*4+j];c[j]=H(neural[i*3+j]);}auto expected=decode(a,b,c,s);for(int j=0;j<3;j++)close(out[i*3+j],expected[j],"decode");
        }
    }
    std::printf("codec failures=%u\n",failures);return failures?1:0;
}
