/* Real Wine process crossing Win64 -> SysV into the rebuilt HIP library. */
#include <windows.h>
#include <stdio.h>
#include <string.h>
int main(void) {
    HMODULE module=LoadLibraryA("dlss5_hip.dll");
    if(!module){printf("LoadLibrary failed=%lu\n",GetLastError());return 1;}
    typedef int (*Run)(const float*,float*,unsigned);
    typedef const char* (*Error)(void);
    Run run=(Run)GetProcAddress(module,"dlss5_run");
    Error error=(Error)GetProcAddress(module,"dlss5_last_error");
    if(!run||!error)return 2;
    int rc=run(NULL,NULL,7);const char*message=error();
    printf("bridge_run=%d diagnostic=%s\n",rc,message?message:"(null)");
    return rc==-1&&message&&strstr(message,"not initialized")?0:3;
}
