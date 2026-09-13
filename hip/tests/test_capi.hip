// Real API error contracts. No fabricated network or preload table.
#include "dlss5_capi.h"
#include <cstdio>
#include <cstring>
int main(){
    dlss5_shutdown();
    if(dlss5_run(nullptr,nullptr,0)!=-1||!std::strstr(dlss5_last_error(),"not initialized"))return 1;
    // Missing paths must be caught inside the C ABI, repeatably.
    for(int i=0;i<2;++i){
        if(dlss5_init("/definitely/missing/dlss5-weights",0)!=-1||!dlss5_last_error()[0])return 2;
        dlss5_shutdown();
    }
    std::printf("C ABI failure/reinitialization: PASS\n");return 0;
}
