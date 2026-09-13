#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/dlss5_capi.h"
int main(void){
    HMODULE module=LoadLibraryA("dlss5_hip.dll");if(!module)return 2;
    typedef int(*Init)(const char*,int);typedef int(*Frame)(const Dlss5Frame*);typedef const char*(*Error)(void);typedef void(*Stop)(void);
    Init init=(Init)GetProcAddress(module,"dlss5_init");Frame run=(Frame)GetProcAddress(module,"dlss5_run_frame");Error error=(Error)GetProcAddress(module,"dlss5_last_error");Stop stop=(Stop)GetProcAddress(module,"dlss5_shutdown");
    if(!init||!run||!error||!stop)return 3;
    if(init(NULL,0)){printf("init: %s\n",error());return 4;}
    size_t n=1920ull*1080;float*rgba=calloc(n*4,sizeof(float)),*rgb=calloc(n*3,sizeof(float));if(!rgba||!rgb)return 5;
    for(size_t i=0;i<n;++i){rgba[i*4]=.2f+(float)(i%37)*.003f;rgba[i*4+1]=.15f;rgba[i*4+2]=.1f;rgba[i*4+3]=1;}
    Dlss5Frame f={0};f.struct_size=sizeof f;f.flags=1;f.rgba=rgba;f.rgb=rgb;f.reset=1;f.paper_white=f.transfer=f.color=1;
    int result=run(&f);size_t changed=0,bad=0;for(size_t i=0;i<n;++i)for(int c=0;c<3;++c){changed+=rgb[i*3+c]!=rgba[i*4+c];bad+=!isfinite(rgb[i*3+c]);}
    printf("Wine frame bridge ABI_size=%zu result=%d changed=%zu nonfinite=%zu error=%s\n",sizeof f,result,changed,bad,error());
    stop();free(rgb);free(rgba);FreeLibrary(module);return result||bad||!changed?1:0;
}
