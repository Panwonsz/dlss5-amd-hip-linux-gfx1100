#include "dlss5_capi.h"
#include <cmath>
#include <cstdio>
#include <vector>
int main(int argc,char**argv){
    if(argc!=2){std::fprintf(stderr,"usage: test_capi_lifecycle WEIGHTS\n");return 2;}
    if(dlss5_init(argv[1],0)){std::fprintf(stderr,"init: %s\n",dlss5_last_error());return 1;}
    // A failed replacement must not destroy a previously working instance.
    if(dlss5_init("/definitely/missing/dlss5-weights",0)!=-1)return 1;
    std::vector<float> in(1920ull*1080*4,.4f),out(1920ull*1080*3,-9.f);
    int result=dlss5_run(in.data(),out.data(),19);
    if(result){std::fprintf(stderr,"lost initialized model: %s\n",dlss5_last_error());return 1;}
    for(float v:out)if(!std::isfinite(v))return 1;
    in[0]=NAN;out.assign(out.size(),-7.f);
    if(dlss5_run(in.data(),out.data(),19)!=-1){std::fprintf(stderr,"V1 accepted NaN input\n");return 1;}
    for(float v:out)if(v!=-7.f)return 1;
    dlss5_shutdown();dlss5_shutdown();
    std::puts("C ABI keeps working model after failed replacement: PASS");
    return 0;
}
