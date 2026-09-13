// Independent scalar products and HLSL exponent maps, not a GPU round trip.
#include "dlss5_launch.hpp"
#include "dlss5_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace dlss5;

static float exp_map(float score, bool vit) {
    float a = vit ? std::fma(score, half_to_float(0x2dbb), 1.708984375f)
                  : std::fma(score, .044921875f, 1.30078125f);
    a = std::clamp(a, vit ? 1.439453125f : 1.03125f, vit ? 1.9775390625f : 1.5693359375f);
    __half h = __float2half(a);
    unsigned short bits;
    std::memcpy(&bits, &h, 2);
    return half_to_float(uint16_t(vit ? (unsigned(bits) << 4) + 0x4000
                                    : (unsigned(bits) << 5) + 0x8000));
}

static int check_attention(bool vit, unsigned tokens) {
    const unsigned c = 32;
    std::vector<float> qkv(tokens * 3 * c), bias(4096);
    for (unsigned t = 0; t < tokens; ++t)
        for (unsigned ch = 0; ch < c; ++ch) {
            qkv[t * 3 * c + ch] = from_e4m3(e4m3_byte(float(int((t*7+ch*3)%13)-6)*.125f));
            qkv[t * 3 * c + c + ch] = from_e4m3(e4m3_byte(float(int((t*3+ch*7)%11)-5)*.125f));
            qkv[t * 3 * c + 2*c + ch] = from_e4m3(e4m3_byte(float(int((t*11+ch*5)%29)-14)*.25f));
        }
    for (unsigned q = 0; q < 64; ++q)
        for (unsigned k = 0; k < 64; ++k)
            bias[q*64+k] = -float((q*3+k*5)%17)*.25f;
    std::vector<u8> ref(tokens*c), out(tokens*c), bytes(qkv.size());
    for (unsigned q=0;q<tokens;++q) {
        std::vector<float> ex(tokens);
        float den=0;
        for(unsigned k=0;k<tokens;++k) {
            double dot=0;
            for(unsigned ch=0;ch<c;++ch)
                dot+=double(qkv[q*3*c+ch])*qkv[k*3*c+c+ch];
            ex[k]=exp_map(float(dot)+(vit?0:bias[q*64+k]),vit);
            den+=ex[k];
        }
        for(unsigned ch=0;ch<c;++ch) {
            double av=0;
            for(unsigned k=0;k<tokens;++k) {
                float p=from_e4m3(e4m3_byte(vit?ex[k]:ex[k]/den));
                av+=double(p)*qkv[k*3*c+2*c+ch];
            }
            ref[q*c+ch]=e4m3_byte(H(float(vit?av/den:av)));
        }
    }
    DeviceBuf dq(qkv.size()*4), db(bias.size()*4), dout(out.size());
    if(vit) {
        dq.upload(qkv.data(),qkv.size()*4);
        launch_vit_attention(dq.as<float>(),dout.as<u8>(),tokens,c);
    } else {
        for(size_t i=0;i<bytes.size();++i) bytes[i]=e4m3_byte(qkv[i]);
        dq.upload(bytes.data(),bytes.size());db.upload(bias.data(),bias.size()*4);
        launch_attention(c,dq.as<u8>(),db.as<float>(),dout.as<u8>(),8,8,1);
    }
    hip_check(hipDeviceSynchronize(),"attention regression");
    dout.download(out.data(),out.size());
    unsigned bad=0;float maxerr=0;
    for(size_t i=0;i<out.size();++i) {
        float actual=from_e4m3(out[i]),expected=from_e4m3(ref[i]);
        if(!std::isfinite(actual)||actual!=expected) ++bad;
        maxerr=std::max(maxerr,std::fabs(actual-expected));
    }
    std::printf("%s attention tokens=%u compared=%zu mismatches=%u max_error=%g\n",
                vit?"ViT":"window",tokens,out.size(),bad,maxerr);
    return bad ? 1 : 0;
}
int main(){return check_attention(false,64)|check_attention(true,16)|check_attention(true,64)|check_attention(true,640);}
