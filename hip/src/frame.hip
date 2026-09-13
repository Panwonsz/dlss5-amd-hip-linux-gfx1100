// Native HIP implementation of the published lmxxf image codecs.
#include "dlss5_frame.hpp"
#include "dlss5_runtime.hpp"
#include <cmath>
namespace dlss5 {
namespace {
struct Vec {float x,y,z;};
__device__ Vec operator+(Vec a,Vec b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
__device__ Vec operator-(Vec a,Vec b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
__device__ Vec operator*(Vec a,float b){return {a.x*b,a.y*b,a.z*b};}
__device__ Vec mix(Vec a,Vec b,float t){return a+(b-a)*t;}
__device__ float sat(float v){return fminf(fmaxf(v,0.f),1.f);}
__device__ float linear(float c){c=sat(c);return c<=.04045f?c/12.92f:powf((c+.055f)/1.055f,2.4f);}
__device__ float srgb(float c){return c<=.0031308f?c*12.92f:1.055f*powf(c,1.f/2.4f)-.055f;}
__device__ Vec matrix(Vec c,const float* m){return {m[0]*c.x+m[1]*c.y+m[2]*c.z,m[3]*c.x+m[4]*c.y+m[5]*c.z,m[6]*c.x+m[7]*c.y+m[8]*c.z};}
__device__ Vec lab(Vec c){
    const float a[]={.4122214708f,.5363325363f,.0514459929f,.2119034982f,.6806995451f,.1073969566f,.0883024619f,.2817188376f,.6299787005f};
    const float b[]={.2104542553f,.7936177850f,-.0040720468f,1.9779984951f,-2.4285922050f,.4505937099f,.0259040371f,.7827717662f,-.8086757660f};
    Vec l=matrix(c,a);return matrix({cbrtf(l.x),cbrtf(l.y),cbrtf(l.z)},b);
}
__device__ Vec unlab(Vec c){
    const float a[]={1,.3963377774f,.2158037573f,1,-.1055613458f,-.0638541728f,1,-.0894841775f,-1.2914855480f};
    const float b[]={4.0767416621f,-3.3077115913f,.2309699292f,-1.2684380046f,2.6097574011f,-.3413193965f,-.0041960863f,-.7034186147f,1.7076147010f};
    Vec l=matrix(c,a);return matrix({l.x*l.x*l.x,l.y*l.y*l.y,l.z*l.z*l.z},b);
}
__device__ Vec ap1(Vec c){
    const float a[]={.613097f,.339523f,.047379f,.070194f,.916354f,.013452f,.020616f,.109570f,.869815f};
    const float b[]={1.705051f,-.621792f,-.083259f,-.130256f,1.140805f,-.010548f,-.024003f,-.128969f,1.152972f};
    Vec l=matrix(c,a);return matrix({fmaxf(l.x,0),fmaxf(l.y,0),fmaxf(l.z,0)},b);
}
__device__ float lum(Vec c){return c.x*.212639f+c.y*.715169f+c.z*.072192f;}
__device__ Vec hue(Vec incorrect,Vec correct){
    Vec a=lab(incorrect),b=lab(correct);float ca=hypotf(a.y,a.z),cb=hypotf(b.y,b.z),ratio=cb==0?1:ca/cb;
    a.y=b.y*ratio;a.z=b.z*ratio;return ap1(unlab(a));
}
__global__ void encode(const float* in,float* out,unsigned n,FrameSettings s){
    unsigned p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;
    for(unsigned c=0;c<3;++c){float v=in[p*4+c];
        if(s.display_srgb)v=sat(v);
        else {v=fmaxf(v,0)/s.paper_white;if(v>.75f)v=1.f-.25f*expf(-5.770780f*(v-.75f));v=srgb(sat(v));}
        out[p*4+c]=H(v);
    }out[p*4+3]=1.f;
}
__global__ void decode(const float* source,const float* proxy,const float* neural,float* out,unsigned n,FrameSettings s){
    unsigned p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;
    Vec original,pr,nr;
    original={source[p*4],source[p*4+1],source[p*4+2]};
    if(s.display_srgb)original={linear(original.x),linear(original.y),linear(original.z)};
    else original={fmaxf(original.x,0)/s.paper_white,fmaxf(original.y,0)/s.paper_white,fmaxf(original.z,0)/s.paper_white};
    pr={linear(proxy[p*4]),linear(proxy[p*4+1]),linear(proxy[p*4+2])};
    nr={linear(H(neural[p*3])),linear(H(neural[p*3+1])),linear(H(neural[p*3+2]))};
    float oy=lum(original),py=lum(pr),ny=lum(nr);Vec upgraded=original;
    if(ny>1e-5f){float ratio=oy<py?oy/fmaxf(py,1e-6f):(ny+fmaxf(0,oy-py))/ny;upgraded=mix(original,hue(nr*ratio,nr),s.transfer);}
    float ratio=oy==0?1:fminf(fmaxf(lum(upgraded)/oy,0),4);
    Vec result=mix(original*ratio,upgraded,s.color);
    if(s.display_srgb)result={srgb(sat(result.x)),srgb(sat(result.y)),srgb(sat(result.z))};
    else result=result*s.paper_white;
    out[p*3]=result.x;out[p*3+1]=result.y;out[p*3+2]=result.z;
}
void validate(unsigned w,unsigned h,FrameSettings s){
    if(!w||!h||w>7680||h>4320||!std::isfinite(s.paper_white)||s.paper_white<=0||
       !std::isfinite(s.transfer)||s.transfer<0||s.transfer>1||!std::isfinite(s.color)||s.color<0||s.color>1)
        throw std::runtime_error("invalid frame codec settings");
}
__device__ float sample_channel(const float* src,unsigned w,unsigned h,unsigned channels,unsigned c,float x,float y){
    x=fminf(fmaxf(x,0.f),float(w-1));y=fminf(fmaxf(y,0.f),float(h-1));
    unsigned x0=unsigned(floorf(x)),y0=unsigned(floorf(y)),x1=min(x0+1,w-1),y1=min(y0+1,h-1);
    float fx=x-x0,fy=y-y0;
    float top=src[(y0*w+x0)*channels+c]*(1-fx)+src[(y0*w+x1)*channels+c]*fx;
    float bottom=src[(y1*w+x0)*channels+c]*(1-fx)+src[(y1*w+x1)*channels+c]*fx;
    return top*(1-fy)+bottom*fy;
}
__device__ void cubic_axis(float p,unsigned n,float* positions,float* weights){
    float center=floorf(p-.5f)+.5f,t=sat(p-center),t2=t*t,t3=t2*t;
    float left=t2-.5f*(t+t3),middle=1.5f*t3-2.5f*t2+1,right=.5f*(t3-t2);
    float other=1-left-middle-right,sum=middle+other;
    positions[0]=fminf(fmaxf(center-1,.5f),float(n)-.5f);
    positions[1]=fminf(fmaxf(center+other/sum,.5f),float(n)-.5f);
    positions[2]=fminf(fmaxf(center+2,.5f),float(n)-.5f);
    weights[0]=left;weights[1]=sum;weights[2]=right;
}
__global__ void warp(const float* current,const float* history,const float* motion,float* out,
    unsigned w,unsigned h,unsigned ow,unsigned oh,unsigned mw,unsigned mh,float sx,float sy,bool reset){
    unsigned p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=ow*oh)return;
    unsigned x=p%ow,y=p/ow,rx=x%(2*(w-1)),ry=y%(2*(h-1));rx=rx<w?rx:2*(w-1)-rx;ry=ry<h?ry:2*(h-1)-ry;
    if(reset){for(unsigned c=0;c<3;++c)out[p*4+c]=current[(ry*w+rx)*4+c];out[p*4+3]=0;return;}
    float ux=(rx+.5f)/w,uy=(ry+.5f)/h;
    float mx=sample_channel(motion,mw,mh,2,0,ux*mw-.5f,uy*mh-.5f);
    float my=sample_channel(motion,mw,mh,2,1,ux*mw-.5f,uy*mh-.5f);
    float px[3],py[3],wx[3],wy[3];cubic_axis(fmaf(mx,sx,ux)*w,w,px,wx);cubic_axis(fmaf(my,sy,uy)*h,h,py,wy);
    float weights[5]={wx[1]*wy[0],wx[0]*wy[1],wx[1]*wy[1],wx[1]*wy[2],wx[2]*wy[1]};
    unsigned ix[5]={1,0,1,1,2},iy[5]={0,1,1,2,1};float total=0;for(int i=0;i<5;++i)total+=weights[i];
    for(unsigned c=0;c<3;++c){float value=0;for(int i=0;i<5;++i)value+=sample_channel(history,w,h,3,c,px[ix[i]]-.5f,py[iy[i]]-.5f)*weights[i];out[p*4+c]=value/total;}out[p*4+3]=1;
}
__global__ void guard(const float* base,float* warped,unsigned n,float dark,float bright){
    unsigned p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;
    if(fmaxf(warped[p*4],fmaxf(warped[p*4+1],warped[p*4+2]))<dark&&fmaxf(base[p*4],fmaxf(base[p*4+1],base[p*4+2]))>bright)
        for(unsigned c=0;c<3;++c)warped[p*4+c]=base[p*4+c];
}
__global__ void smooth(const float* warped,float* rgb,unsigned n,float threshold,float strength){
    unsigned p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n||warped[p*4+3]<=0)return;
    float d=0;for(unsigned c=0;c<3;++c)d=fmaxf(d,fabsf(rgb[p*3+c]-warped[p*4+c]));
    float weight=strength*sat(1-d/threshold);for(unsigned c=0;c<3;++c)rgb[p*3+c]+=weight*(warped[p*4+c]-rgb[p*3+c]);
}
}
void frame_encode(const float* in,float* out,unsigned w,unsigned h,FrameSettings s,hipStream_t stream){
    validate(w,h,s);if(!in||!out||in==out)throw std::runtime_error("codec input/alias");
    encode<<<(w*h+255)/256,256,0,stream>>>(in,out,w*h,s);hip_check(hipGetLastError(),"encode launch");
}
void frame_decode(const float* in,const float* proxy,const float* neural,float* out,unsigned w,unsigned h,FrameSettings s,hipStream_t stream){
    validate(w,h,s);if(!in||!proxy||!neural||!out||out==in||out==proxy||out==neural)throw std::runtime_error("codec output/alias");
    decode<<<(w*h+255)/256,256,0,stream>>>(in,proxy,neural,out,w*h,s);hip_check(hipGetLastError(),"decode launch");
}
void frame_warp(const float* current,const float* history,const float* motion,float* out,unsigned w,unsigned h,unsigned ow,unsigned oh,
                 unsigned mw,unsigned mh,float sx,float sy,bool reset,hipStream_t stream){
    validate(w,h,FrameSettings{});
    if(w<2||h<2||ow<w||oh<h||ow>8192||oh>8192||!current||!out||out==current||out==history||out==motion||!std::isfinite(sx)||!std::isfinite(sy))throw std::runtime_error("invalid temporal geometry/alias");
    reset=reset||!history||!motion;
    if(!reset&&(!mw||!mh||mw>7680||mh>4320))throw std::runtime_error("invalid motion geometry");
    warp<<<(ow*oh+255)/256,256,0,stream>>>(current,history,motion,out,w,h,ow,oh,mw,mh,sx,sy,reset);hip_check(hipGetLastError(),"warp launch");
}
void frame_guard(const float* base,float* out,unsigned n,float dark,float bright,hipStream_t stream){
    if(!base||!out||!n||!std::isfinite(dark)||!std::isfinite(bright)||dark<0||bright<0)throw std::runtime_error("invalid history guard");
    guard<<<(n+255)/256,256,0,stream>>>(base,out,n,dark,bright);hip_check(hipGetLastError(),"guard launch");
}
void frame_smooth(const float* warp,float* rgb,unsigned n,float threshold,float strength,hipStream_t stream){
    if(!warp||!rgb||!n||!std::isfinite(threshold)||threshold<=0||!std::isfinite(strength)||strength<0||strength>.8f)throw std::runtime_error("invalid history smooth");
    smooth<<<(n+255)/256,256,0,stream>>>(warp,rgb,n,threshold,strength);hip_check(hipGetLastError(),"smooth launch");
}
} // namespace dlss5
