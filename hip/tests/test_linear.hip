#include "dlss5_launch.hpp"
#include "dlss5_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace dlss5;
int main() {
    const uint m=32,n=64,k=96;
    std::vector<float> in(m*k),w(n*k),skip(m*n),sc(n),out(m*n);
    for(size_t i=0;i<in.size();++i) in[i]=float(int(i%23)-11)*.137f;
    for(size_t i=0;i<w.size();++i) w[i]=from_e4m3(e4m3_byte(float(int((i*3+i/19)%17)-8)*.03125f));
    for(size_t i=0;i<skip.size();++i) skip[i]=float(int(i%13)-6)*.211f;
    for(size_t i=0;i<sc.size();++i) sc[i]=H(float(int(i%11)-5)*.073f);
    DeviceBuf di(in.size()*4),dw=upload_tiled_weights(w.data(),n,k),ds(skip.size()*4),dc(sc.size()*4),dout(out.size()*4);
    di.upload(in.data(),in.size()*4);ds.upload(skip.data(),skip.size()*4);dc.upload(sc.data(),sc.size()*4);
    unsigned bad=0;
    for(int mode=0;mode<5;++mode){
        launch_linear_f32(di.as<float>(),dw.as<u8>(),dout.as<float>(),m,n,k,
                          mode==2?nullptr:ds.as<float>(),mode==2?nullptr:dc.as<float>(),mode,0);
        hip_check(hipDeviceSynchronize(),"linear");dout.download(out.data(),out.size()*4);
        float maxerr=0;
        for(uint row=0;row<m;++row)for(uint col=0;col<n;++col){
            double acc=(mode==2||mode==4)?0:H(skip[row*n+col]*sc[col]);
            for(uint ch=0;ch<k;++ch)acc+=double(from_e4m3(e4m3_byte(in[row*k+ch])))*w[col*k+ch];
            float v=float(acc),expected;
            if(mode==4)v+=skip[row*n+col]*sc[col];
            if(mode==2)expected=from_e4m3(e4m3_byte(ActivatePoly(v)));
            else if(mode==1)expected=from_e4m3(e4m3_byte(H(v)));
            else if(mode==3)expected=v;
            else expected=H(v);
            float err=std::fabs(out[row*n+col]-expected);
            if(!std::isfinite(out[row*n+col])||err>1e-5f)++bad;
            maxerr=std::max(maxerr,err);
        }
        std::printf("linear mode=%d max_error=%g\n",mode,maxerr);
    }
    std::printf("linear mismatches=%u\n",bad);
    return bad?1:0;
}
