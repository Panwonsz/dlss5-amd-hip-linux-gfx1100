#include "dlss5_launch.hpp"
#include "dlss5_runtime.hpp"
#include <cstdio>
#include <vector>
using namespace dlss5;
int main() {
    constexpr uint c=64,n=64;
    std::vector<u8> x(n*c), out(n*c), zeros(8*c*c);
    for (size_t i=0;i<x.size();++i) x[i]=e4m3_byte(float(int(i%19)-9)*.25f);
    std::vector<float> scale(c,1.f),bias(2*4096,0.f);
    DeviceBuf dx(x.size()),df(zeros.size()),dp(c*c),dq(3*c*c),ds(c*4),db(bias.size()*4),qtmp(n*3*c),atmp(n*c),dy(n*c);
    dx.upload(x.data(),x.size());df.zero();dp.zero();dq.zero();ds.upload(scale.data(),scale.size()*4);db.upload(bias.data(),bias.size()*4);
    launch_fused_mh(c,dx.as<u8>(),df.as<u8>(),dp.as<u8>(),ds.as<float>(),dq.as<u8>(),db.as<float>(),dp.as<u8>(),ds.as<float>(),qtmp.as<u8>(),atmp.as<u8>(),dy.as<u8>(),8,8,1,1);
    hip_check(hipDeviceSynchronize(),"residual");dy.download(out.data(),out.size());
    unsigned bad=0;for(size_t i=0;i<x.size();++i)if(from_e4m3(x[i])!=from_e4m3(out[i]))++bad;
    std::printf("zero branches preserve learned skips: compared=%zu mismatches=%u\n",x.size(),bad);
    return bad ? 1 : 0;
}
