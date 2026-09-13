// Real MinGW header/link probe. Not a game or a fake HIP execution.
#include "../../src/native_hip_live.h"
extern "C" __declspec(dllexport) bool ProbeNativeHipLive(
    NativeHipLive* live,ID3D12GraphicsCommandList* list,ID3D12Resource* color,
    D3D12_RESOURCE_STATES state,uint64_t frame_id){
    return live && live->Record(list,color,state,frame_id);
}
int main(){NativeHipLive live;return live.Record(nullptr,nullptr,D3D12_RESOURCE_STATE_COMMON,0)?1:0;}
