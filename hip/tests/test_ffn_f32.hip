#include "dlss5_launch.hpp"
#include "dlss5_runtime.hpp"
#include "ulp_compare.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace dlss5;
int main(){
    UlpTally tally;
    for(unsigned c:{32u,64u,128u,256u}){
        const unsigned n=32,hidden=4*c;
        std::vector<float> x(n*c),w(8*c*c),sc(c),out(n*c);
        for(size_t i=0;i<x.size();++i)x[i]=H(float(int((i*7)%17)-8)*.073f);
        for(size_t i=0;i<w.size();++i)w[i]=float(int((i*11+i/37)%7)-3)*.03125f;
        for(size_t i=0;i<sc.size();++i)sc[i]=H(float(int(i%13)-6)*.077f);
        std::vector<u8> tiled(w.size());
        pack_tiled_e4m3(tiled.data(),w.data(),hidden,c);
        pack_tiled_e4m3(tiled.data()+hidden*c,w.data()+hidden*c,c,hidden);
        DeviceBuf dx(x.size()*4),dw(tiled.size()),ds(sc.size()*4),dy(out.size()*4);
        dx.upload(x.data(),x.size()*4);dw.upload(tiled.data(),tiled.size());ds.upload(sc.data(),sc.size()*4);
        launch_ffn_f32(c,dx.as<float>(),dw.as<u8>(),c==32?ds.as<float>():nullptr,dy.as<float>(),n);
        hip_check(hipDeviceSynchronize(),"ffn f32");dy.download(out.data(),out.size()*4);
        float maxerr=0;
        for(unsigned t=0;t<n;++t){
            std::vector<float> mid(hidden);
            for(unsigned j=0;j<hidden;++j){float a=0;for(unsigned k=0;k<c;++k)a+=from_e4m3(e4m3_byte(x[t*c+k]))*w[j*c+k];mid[j]=from_e4m3(e4m3_byte(ActivatePoly(a)));}
            for(unsigned j=0;j<c;++j){float a=c==32?x[t*c+j]*sc[j]:0;
                for(unsigned k=0;k<hidden;++k)a+=mid[k]*w[hidden*c+j*hidden+k];
                // C32 ends in H(), the wider widths quantize to E4M3; one step is measured in
                // whichever grid the value actually lands on.
                float e=c==32?H(a):from_e4m3(e4m3_byte(a));
                tally.check(out[t*c+j],e,c==32?Grid::F16:Grid::E4M3);
                maxerr=std::max(maxerr,std::fabs(e-out[t*c+j]));
            }
        }
        std::printf("FFN C%u max_error=%g\n",c,maxerr);
    }
    return tally.report("FFN");
}
