#include "dlss5_frame.hpp"
#include "dlss5_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace dlss5;
static unsigned bad;
static void same(float a,float b,const char*n){if(!std::isfinite(a)||std::fabs(a-b)>2e-6f){if(bad<5)std::printf("%s got=%g expected=%g\n",n,a,b);++bad;}}
int main(){
    const unsigned w=16,h=8,ow=24,oh=16,mw=8,mh=4;
    std::vector<float> history(w*h*3),current(w*h*4),motion(mw*mh*2),out(ow*oh*4);
    for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){for(unsigned c=0;c<3;++c){history[(y*w+x)*3+c]=float(2*x+3*y+c)/128;current[(y*w+x)*4+c]=.75f;}current[(y*w+x)*4+3]=1;}
    DeviceBuf dh(history.size()*4),dc(current.size()*4),dm(motion.size()*4),doo(out.size()*4);
    dh.upload(history.data(),history.size()*4);dc.upload(current.data(),current.size()*4);
    auto reflect=[](unsigned p,unsigned n){unsigned r=p%(2*(n-1));return r<n?r:2*(n-1)-r;};
    for(int reset=0;reset<2;++reset)for(float dx:{0.f,1.f}){
        for(unsigned i=0;i<mw*mh;++i){motion[i*2]=dx;motion[i*2+1]=0;}
        dm.upload(motion.data(),motion.size()*4);
        frame_warp(dc.as<float>(),dh.as<float>(),dm.as<float>(),doo.as<float>(),w,h,ow,oh,mw,mh,1.f/w,1.f/h,reset,nullptr);
        hip_check(hipDeviceSynchronize(),"warp");doo.download(out.data(),out.size()*4);
        for(unsigned y=0;y<oh;++y)for(unsigned x=0;x<ow;++x){unsigned sx=std::min(reflect(x,w)+unsigned(dx),w-1),sy=reflect(y,h);
            for(unsigned c=0;c<3;++c)same(out[(y*ow+x)*4+c],reset?.75f:history[(sy*w+sx)*3+c],"warp/reset");same(out[(y*ow+x)*4+3],reset?0:1,"valid-alpha");}
    }
    // Fractional displacement on an affine ramp is reproduced by the cubic filter away from edges.
    for(unsigned i=0;i<mw*mh;++i){motion[i*2]=.25f;motion[i*2+1]=.5f;}
    dm.upload(motion.data(),motion.size()*4);
    frame_warp(dc.as<float>(),dh.as<float>(),dm.as<float>(),doo.as<float>(),w,h,ow,oh,mw,mh,1.f/w,1.f/h,false,nullptr);
    hip_check(hipDeviceSynchronize(),"fractional");doo.download(out.data(),out.size()*4);
    // Independent five-tap cross weights (not the ordinary bicubic 16-tap filter).
    auto axis=[](float t,float&left,float&middle,float&right,float&position){left=t*t-.5f*(t+t*t*t);float center=1.5f*t*t*t-2.5f*t*t+1;right=.5f*(t*t*t-t*t);float other=1-left-center-right;middle=center+other;position=other/middle;};
    float lx,cx,rx,px,ly,cy,ry,py;axis(.25f,lx,cx,rx,px);axis(.5f,ly,cy,ry,py);
    float total=cx*ly+lx*cy+cx*cy+cx*ry+rx*cy;
    for(unsigned y=2;y<h-3;++y)for(unsigned x=2;x<w-3;++x)for(unsigned c=0;c<3;++c){
        auto f=[&](float xx,float yy){return (2*xx+3*yy+c)/128;};
        float e=(f(x+px,y-1.f)*cx*ly+f(x-1.f,y+py)*lx*cy+f(x+px,y+py)*cx*cy+f(x+px,y+2.f)*cx*ry+f(x+2.f,y+py)*rx*cy)/total;
        same(out[(y*ow+x)*4+c],e,"fractional cubic");}
    // Guard compares strict max RGB thresholds and keeps alpha.
    std::vector<float> base(4*4,.75f),warp(4*4,.001f),rgb(4*3,.8f);
    warp[3]=1;warp[7]=0;warp[11]=1;warp[15]=1;warp[8]=.02f;warp[12]=.3f;
    DeviceBuf db(base.size()*4),dw(warp.size()*4),dr(rgb.size()*4);db.upload(base.data(),base.size()*4);dw.upload(warp.data(),warp.size()*4);dr.upload(rgb.data(),rgb.size()*4);
    frame_guard(db.as<float>(),dw.as<float>(),4,.01f,.1f,nullptr);hip_check(hipDeviceSynchronize(),"guard");dw.download(warp.data(),warp.size()*4);
    same(warp[0],.75f,"guard-dark");same(warp[3],1,"guard-alpha");same(warp[8],.02f,"guard-notdark");
    frame_smooth(dw.as<float>(),dr.as<float>(),4,.1f,.8f,nullptr);hip_check(hipDeviceSynchronize(),"smooth");dr.download(rgb.data(),rgb.size()*4);
    same(rgb[0],.78f,"smooth");same(rgb[3],.8f,"invalid-history");same(rgb[6],.8f,"large-change");
    std::printf("temporal failures=%u\n",bad);return bad?1:0;
}
