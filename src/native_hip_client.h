#pragma once
/* Windows add-on client for the HIP network. Loads dlss5_hip.dll (PE trampoline to libdlss5_hip.so). */
#include "native_lab_paths.h"
#include <windows.h>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <mutex>
#include "../hip/include/dlss5_capi.h"

class NativeHipClient {
 using InitFn=int(*)(const char*,int);
 using RunFn=int(*)(const float*,float*,unsigned);
 using FrameFn=int(*)(const Dlss5Frame*);
 using ErrFn=const char*(*)();
 using ShutFn=void(*)();
 HMODULE dll{};InitFn p_init{};RunFn p_run{};ErrFn p_err{};ShutFn p_shut{};bool ready_{};
 FrameFn p_frame{};bool initialized_{};
 // One reservation across instances, including loading/initializing/cleanup.
 // The same mutex serializes calls so shutdown cannot race a wrapper run.
 static std::mutex& Mutex(){static std::mutex mutex;return mutex;}
 static NativeHipClient*& Owner(){static NativeHipClient* owner=nullptr;return owner;}
 void Release(){ // Mutex held; failed init never grants shutdown ownership.
  if(initialized_&&p_shut)p_shut();
  ready_=false;initialized_=false;
  if(dll)FreeLibrary(dll);
  dll=nullptr;p_init=nullptr;p_run=nullptr;p_frame=nullptr;p_err=nullptr;p_shut=nullptr;
  if(Owner()==this)Owner()=nullptr;
 }
 static void Log(const char*e,const char*d=""){
  if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-game-oneshot.txt").c_str(),L"ab")){
   fprintf(f,"pid=%lu tick=%llu event=%s detail=%s\n",GetCurrentProcessId(),GetTickCount64(),e,d);fclose(f);}
 }
public:
 NativeHipClient()=default;
 NativeHipClient(const NativeHipClient&)=delete;
 NativeHipClient& operator=(const NativeHipClient&)=delete;
 NativeHipClient(NativeHipClient&&)=delete;
 NativeHipClient& operator=(NativeHipClient&&)=delete;
 bool Ready()const{std::lock_guard<std::mutex> lock(Mutex());return ready_;}
 void Create(const char* adapter_name=nullptr){
  std::lock_guard<std::mutex> lock(Mutex());
  if(Owner())throw std::runtime_error("HIP client already owned; destroy the active client before Create");
  Owner()=this;
  try{
  wchar_t path[MAX_PATH]{};HMODULE self=nullptr;
  GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                     reinterpret_cast<LPCWSTR>(&NativeLabRoot),&self);
  GetModuleFileNameW(self,path,MAX_PATH);
  std::wstring dir(path);size_t slash=dir.find_last_of(L"\\/");if(slash!=std::wstring::npos)dir.resize(slash);
  std::wstring dll_path=dir+L"\\dlss5_hip.dll";
  dll=LoadLibraryW(dll_path.c_str());
  if(!dll){Log("hip_load_failed","dlss5_hip.dll");throw std::runtime_error("dlss5_hip.dll missing (HIP trampoline)");}
  p_init=reinterpret_cast<InitFn>(GetProcAddress(dll,"dlss5_init"));
  p_run=reinterpret_cast<RunFn>(GetProcAddress(dll,"dlss5_run"));
  p_frame=reinterpret_cast<FrameFn>(GetProcAddress(dll,"dlss5_run_frame"));
  p_err=reinterpret_cast<ErrFn>(GetProcAddress(dll,"dlss5_last_error"));
  p_shut=reinterpret_cast<ShutFn>(GetProcAddress(dll,"dlss5_shutdown"));
  if(!p_init||!p_run||!p_frame||!p_err||!p_shut)throw std::runtime_error("dlss5_hip.dll exports missing; rebuild the HIP bridge");
  int gpu=0;
  if(adapter_name){
   using FindFn=int(*)(const char*);
   auto find=reinterpret_cast<FindFn>(GetProcAddress(dll,"dlss5_find_device"));
   if(!find)throw std::runtime_error("HIP adapter matching export missing");
   gpu=find(adapter_name);
   if(gpu<0)throw std::runtime_error(std::string("HIP adapter: ")+p_err());
   Log("hip_adapter",adapter_name);
  }
  int rc=p_init(nullptr,gpu); /* Linux side reads DLSS5_HIP_WEIGHTS */
  if(rc){const char*e=p_err?p_err():"init failed";Log("hip_init_failed",e);throw std::runtime_error(std::string("HIP init: ")+e);}
  initialized_=true; // Only successful init permits shutdown of this model.
  // V2 handshake (API-4): after successful init, null frame must return -1
  // with the host's exact validation diagnostic. Missing V2 must instead set
  // the PE-local "hip frame bridge missing ..." diagnostic. Fail closed for
  // old/unknown bridges. This cannot infer or allocate frame/GPU work, and is
  // deliberately AFTER init: probing earlier could invalidate an independent
  // model's history when our init subsequently fails.
  rc=p_frame(nullptr);const char*frame_error=p_err();
  if(rc!=-1||!frame_error||std::strcmp(frame_error,"invalid frame ABI/buffers"))
   throw std::runtime_error(std::string("HIP V2 handshake: ")+(frame_error?frame_error:"missing diagnostic"));
  ready_=true;Log("hip_ready","libdlss5_hip.so");
  }catch(...){Release();throw;}
 }
 int Run(const float*rgba,float*rgb,unsigned seed){
  std::lock_guard<std::mutex> lock(Mutex());
  if(!ready_||!p_run)return -1;
  int rc=p_run(rgba,rgb,seed);
  if(rc&&p_err)Log("hip_run_failed",p_err());
  return rc;
 }
 int RunFrame(const float*rgba,float*rgb,unsigned seed,bool display_srgb){
  std::lock_guard<std::mutex> lock(Mutex());
  if(!ready_||!p_frame)return -1;
  Dlss5Frame f{};f.struct_size=sizeof f;f.flags=display_srgb?1:0;f.rgba=rgba;f.rgb=rgb;
  f.seed=seed;f.reset=1;f.paper_white=f.transfer=f.color=1;
  // The D3D bridge does not invent missing motion vectors. Host-side temporal
  // callers use run_frame with explicit XY buffers; this path resets history.
  int rc=p_frame(&f);if(rc&&p_err)Log("hip_run_failed",p_err());return rc;
 }
 ~NativeHipClient(){std::lock_guard<std::mutex> lock(Mutex());Release();}
};

#include "native_frame_input_check.h"

inline constexpr unsigned NativeHipNetWidth=1920,NativeHipNetHeight=1080;

inline void NativeResizeChannels(const float*src,unsigned sw,unsigned sh,float*dst,unsigned dw,unsigned dh,unsigned ch){
 if(!src||!dst||!sw||!sh||!dw||!dh||!ch)return;
 if(sw==dw&&sh==dh){std::memcpy(dst,src,size_t(dw)*dh*ch*sizeof(float));return;}
 for(unsigned y=0;y<dh;y++){
  float fy=(y+0.5f)*float(sh)/float(dh)-0.5f;
  int y0=int(floorf(fy)),y1=y0+1;float ty=fy-float(y0);
  if(y0<0){y0=y1=0;ty=0.f;}else if(y0>=int(sh-1)){y0=y1=int(sh-1);ty=0.f;}
  for(unsigned x=0;x<dw;x++){
   float fx=(x+0.5f)*float(sw)/float(dw)-0.5f;
   int x0=int(floorf(fx)),x1=x0+1;float tx=fx-float(x0);
   if(x0<0){x0=x1=0;tx=0.f;}else if(x0>=int(sw-1)){x0=x1=int(sw-1);tx=0.f;}
   const float*p00=src+(size_t(unsigned(y0))*sw+unsigned(x0))*ch;
   const float*p10=src+(size_t(unsigned(y0))*sw+unsigned(x1))*ch;
   const float*p01=src+(size_t(unsigned(y1))*sw+unsigned(x0))*ch;
   const float*p11=src+(size_t(unsigned(y1))*sw+unsigned(x1))*ch;
   float*o=dst+(size_t(y)*dw+x)*ch;
   for(unsigned c=0;c<ch;c++)
    o[c]=(1.f-tx)*(1.f-ty)*p00[c]+tx*(1.f-ty)*p10[c]+(1.f-tx)*ty*p01[c]+tx*ty*p11[c];
  }
 }
}

// Pixel conversion lives in the portable native_frame_input_check.h.
// Callers must supply NativePixelFormat and check the bool result.

inline bool NativeHipRequested(){
 const wchar_t*v=_wgetenv(L"DLSS5_HIP");return v&&!wcscmp(v,L"1");
}
