#pragma once
// Record ONLY on the original FFX native list, immediately after successful
// ffxDispatch. No submission, list lifecycle changes, or graphics waits here.
#include <d3d12.h>
#include <dxgi1_4.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <limits>
#include "native_hip_client.h"
#include "native_device_identity.h"
#include "../hip/include/dlss5_submit.h"

class NativeHipLive {
    template<class T> struct Ref {
        T* p{};
        Ref()=default;
        explicit Ref(T* v):p(v){if(p)p->AddRef();}
        Ref(const Ref&)=delete;
        Ref& operator=(const Ref&)=delete;
        ~Ref(){if(p)p->Release();}
        T** Out(){return &p;}
        T* operator->()const{return p;}
    };
    static void Log(const char* event,uint64_t frame=0,const char* detail="",size_t changed=0) noexcept {
        try {
            static std::mutex mutex;
            std::lock_guard<std::mutex> lock(mutex);
            CreateDirectoryW(NativeLabPath(L"logs").c_str(),nullptr);
            if(FILE* f=_wfopen(NativeLabPath(L"logs\\native-hip-live.txt").c_str(),L"ab")){
                fprintf(f,"pid=%lu frame_id=%llu event=%s changed_rgb_bytes=%llu detail=%s\n",
                    GetCurrentProcessId(),static_cast<unsigned long long>(frame),event,
                    static_cast<unsigned long long>(changed),detail);
                fclose(f);
            }
        }catch(...){}
    }
    struct State {
        Ref<ID3D12Device> device;
        std::shared_ptr<NativeHipClient> client;
        std::atomic<int> init{0}; // 0 cold, 1 initializing, 2 ready, -1 failed
        std::atomic<bool> disabled{false},key_down{false};
        std::atomic<uint64_t> generation{0},queue{0}; // even generation = enabled
        std::atomic<unsigned> inflight{0};
        std::mutex recording,processing;
        bool display_srgb{};
    };
    std::shared_ptr<State> state_=std::make_shared<State>();
    static constexpr unsigned MaxInflight=3;
    // This anchor is needed even on a runtime that ignores SetMarker: prefix
    // copy resources must not die while the recorded list can still execute.
    // Rejected markers stay anchored until LIST DESTRUCTION, conservatively
    // (D3D12 has no allocator-reset notification available to this caller).
    inline static constexpr GUID AnchorId={0xda06bd21,0x147c,0x467a,{0x89,0xd3,0x6d,0xc0,0xd8,0xa9,0x64,0xd3}};
    static GUID NextAnchor(){
        static std::atomic<uint64_t> serial{0};
        const uint64_t n=serial.fetch_add(1);GUID id=AnchorId;
        id.Data1+=uint32_t(n);id.Data2+=uint16_t(n>>32);id.Data3+=uint16_t(n>>48);return id;
    }
    struct DeviceResources {
        Ref<ID3D12Device> device;
        Ref<ID3D12Resource> color,readback,upload;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        UINT64 bytes{};
        unsigned char* mapped{};
        ~DeviceResources(){if(mapped&&upload.p)upload->Unmap(0,nullptr);}
        bool Create(ID3D12Device* d,ID3D12Resource* c,const D3D12_RESOURCE_DESC& desc,size_t bpp){
            device.p=d;d->AddRef();color.p=c;c->AddRef();
            UINT rows=0;UINT64 rowbytes=0;
            d->GetCopyableFootprints(&desc,0,1,0,&fp,&rows,&rowbytes,&bytes);
            const UINT64 packed=desc.Width*bpp;
            if(rows!=desc.Height||rowbytes!=packed||fp.Footprint.Width!=desc.Width||
               fp.Footprint.Height!=desc.Height||fp.Footprint.Depth!=1||fp.Footprint.Format!=desc.Format||
               fp.Footprint.RowPitch<packed||!bytes||bytes>std::numeric_limits<SIZE_T>::max()||
               fp.Offset>bytes||UINT64(rows-1)*fp.Footprint.RowPitch+packed>bytes-fp.Offset)return false;
            D3D12_RESOURCE_DESC buffer{};buffer.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
            buffer.Width=bytes;buffer.Height=1;buffer.DepthOrArraySize=buffer.MipLevels=1;
            buffer.SampleDesc.Count=1;buffer.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            D3D12_HEAP_PROPERTIES heap{};heap.CreationNodeMask=heap.VisibleNodeMask=1;
            heap.Type=D3D12_HEAP_TYPE_READBACK;
            if(FAILED(d->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&buffer,
                D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(readback.Out()))))return false;
            heap.Type=D3D12_HEAP_TYPE_UPLOAD;
            if(FAILED(d->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&buffer,
                D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(upload.Out()))))return false;
            D3D12_RANGE none{};void* ptr=nullptr;
            if(FAILED(upload->Map(0,&none,&ptr))||!ptr)return false;
            mapped=static_cast<unsigned char*>(ptr);return true;
        }
    };
    struct Job final: IUnknown {
        std::atomic<ULONG> refs{1};
        std::atomic<bool> called{false};
        const std::shared_ptr<State> owner;
        const uint64_t frame,generation;
        const GUID anchor=NextAnchor();
        const NativePixelFormat format;
        const unsigned width,height;
        DeviceResources resources;
        Job(std::shared_ptr<State> s,uint64_t f,uint64_t g,NativePixelFormat p,unsigned w,unsigned h):
            owner(std::move(s)),frame(f),generation(g),format(p),width(w),height(h){}
        ~Job(){owner->inflight.fetch_sub(1);}
        HRESULT STDMETHODCALLTYPE QueryInterface(REFIID id,void** out) override {
            if(!out)return E_POINTER;
            *out=nullptr;if(!(id==IID_IUnknown))return E_NOINTERFACE;
            *out=static_cast<IUnknown*>(this);AddRef();return S_OK;
        }
        ULONG STDMETHODCALLTYPE AddRef() override{return refs.fetch_add(1)+1;}
        ULONG STDMETHODCALLTYPE Release() override{ULONG n=refs.fetch_sub(1)-1;if(!n)delete this;return n;}
        static void DLSS5_PE_ABI Retain(void* p) noexcept {static_cast<Job*>(p)->AddRef();}
        static void DLSS5_PE_ABI ReleaseContext(void* p) noexcept {static_cast<Job*>(p)->Release();}
        static int DLSS5_PE_ABI Run(void* p,uint64_t queue) noexcept {
            auto& j=*static_cast<Job*>(p);
            try{return j.Process(queue);}catch(...){Log("callback_failure",j.frame,"device must be lost; suffix not safe");return -1;}
        }
        int Process(uint64_t queue){
            if(called.exchange(true)){Log("duplicate_callback",frame,"resubmitted command list rejected");return -1;}
            uint64_t expected=0;
            if(!queue||(!owner->queue.compare_exchange_strong(expected,queue)&&expected!=queue)){
                Log("queue_conflict",frame,"callback rejected before readback");return -1;
            }
            std::unique_lock<std::mutex> serial(owner->processing,std::try_to_lock);
            if(!serial.owns_lock())return -1;
            if(FAILED(resources.device->GetDeviceRemovedReason()))return -1;
            void* input=nullptr;D3D12_RANGE read{SIZE_T(resources.fp.Offset),SIZE_T(resources.bytes)};
            if(FAILED(resources.readback->Map(0,&read,&input))||!input)return -1;
            struct Unmap {ID3D12Resource* p;~Unmap(){D3D12_RANGE none{};p->Unmap(0,&none);}} unmap{resources.readback.p};
            const size_t row=size_t(width)*NativePixelBytes(format);
            for(unsigned y=0;y<height;y++)std::memcpy(resources.mapped+resources.fp.Offset+size_t(y)*resources.fp.Footprint.RowPitch,
                static_cast<const unsigned char*>(input)+resources.fp.Offset+size_t(y)*resources.fp.Footprint.RowPitch,row);
            // Upload is now a complete original-frame fallback before ANY
            // allocation/conversion/HIP work that might fail. Never reuse output.
            auto bypass=[&]{return (generation&1)||owner->disabled.load()||owner->generation.load()!=generation;};
            if(bypass()){Log("processed",frame,"F6 bypass; original input");return 0;}
            try {
                std::vector<unsigned char> packed(row*height);
                for(unsigned y=0;y<height;y++)std::memcpy(packed.data()+size_t(y)*row,
                    static_cast<const unsigned char*>(input)+resources.fp.Offset+size_t(y)*resources.fp.Footprint.RowPitch,row);
                std::vector<float> rgba;
                if(!NativePixelsToRgbaF32(packed.data(),format,width,height,rgba)){
                    Log("processed",frame,"input rejected; original input");return 0;
                }
                std::vector<float> net_rgba(size_t(NativeHipNetWidth)*NativeHipNetHeight*4);
                std::vector<float> net_rgb(size_t(NativeHipNetWidth)*NativeHipNetHeight*3);
                NativeResizeChannels(rgba.data(),width,height,net_rgba.data(),NativeHipNetWidth,NativeHipNetHeight,4);
                if(bypass()){Log("processed",frame,"F6 bypass; original input");return 0;}
                if(owner->client->RunFrame(net_rgba.data(),net_rgb.data(),unsigned(frame),owner->display_srgb)){
                    Log("processed",frame,"HIP failed; original input");return 0;
                }
                std::vector<float> rgb(size_t(width)*height*3);
                NativeResizeChannels(net_rgb.data(),NativeHipNetWidth,NativeHipNetHeight,rgb.data(),width,height,3);
                if(!NativeRgbF32ToPixels(rgb.data(),packed.data(),format,width,height)){
                    Log("processed",frame,"output rejected; original input");return 0;
                }
                if(bypass()){Log("processed",frame,"F6 changed since Record; original input");return 0;}
                size_t changed=0;const size_t bpp=NativePixelBytes(format),rgbbytes=bpp/4*3;
                for(unsigned y=0;y<height;y++){
                    auto* dst=resources.mapped+resources.fp.Offset+size_t(y)*resources.fp.Footprint.RowPitch;
                    auto* src=packed.data()+size_t(y)*row;
                    // UPLOAD can be write-combined VRAM. CPU reads from it
                    // serialize PCIe transactions; compare cached READBACK instead.
                    const auto* original=static_cast<const unsigned char*>(input)+resources.fp.Offset+size_t(y)*resources.fp.Footprint.RowPitch;
                    for(size_t x=0;x<row;x+=bpp)for(size_t c=0;c<rgbbytes;c++)changed+=original[x+c]!=src[x+c];
                    std::memcpy(dst,src,row);
                }
                // If a recording thread toggled during upload encoding, roll
                // back from this job's completed readback, including exact alpha.
                if(bypass()){
                    for(unsigned y=0;y<height;y++)std::memcpy(resources.mapped+resources.fp.Offset+size_t(y)*resources.fp.Footprint.RowPitch,
                        static_cast<const unsigned char*>(input)+resources.fp.Offset+size_t(y)*resources.fp.Footprint.RowPitch,row);
                    Log("processed",frame,"F6 changed during encoding; original input");return 0;
                }
                Log("processed",frame,"live HIP network; reset=1 (not temporal)",changed);return 0;
            }catch(...){Log("processed",frame,"HIP/conversion exception; original input");return 0;}
        }
    };
    static bool ValidState(D3D12_RESOURCE_STATES s){
        switch(s){
        case D3D12_RESOURCE_STATE_COMMON:case D3D12_RESOURCE_STATE_RENDER_TARGET:
        case D3D12_RESOURCE_STATE_UNORDERED_ACCESS:case D3D12_RESOURCE_STATE_COPY_SOURCE:
        case D3D12_RESOURCE_STATE_COPY_DEST:case D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:
        case D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE:
        case static_cast<D3D12_RESOURCE_STATES>(int(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)|int(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)):return true;
        default:return false;
        }
    }
    static void Copy(ID3D12GraphicsCommandList* list,Job& job,D3D12_RESOURCE_STATES state,bool suffix){
        auto& r=job.resources;
        D3D12_RESOURCE_STATES transfer=suffix?D3D12_RESOURCE_STATE_COPY_DEST:D3D12_RESOURCE_STATE_COPY_SOURCE;
        D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition={r.color.p,0,state,transfer};
        if(state!=transfer)list->ResourceBarrier(1,&b);
        D3D12_TEXTURE_COPY_LOCATION texture{},buffer{};
        texture.pResource=r.color.p;texture.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        buffer.pResource=suffix?r.upload.p:r.readback.p;buffer.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        buffer.PlacedFootprint=r.fp;
        list->CopyTextureRegion(suffix?&texture:&buffer,0,0,0,suffix?&buffer:&texture,nullptr);
        std::swap(b.Transition.StateBefore,b.Transition.StateAfter);
        if(state!=transfer)list->ResourceBarrier(1,&b);
    }
    static void Initialize(std::shared_ptr<State> s) noexcept {
        try {
            Ref<IDXGIFactory4> factory;Ref<IDXGIAdapter1> adapter;
            DXGI_ADAPTER_DESC1 desc{};
            if(FAILED(CreateDXGIFactory1(IID_PPV_ARGS(factory.Out())))||
               FAILED(factory->EnumAdapterByLuid(s->device->GetAdapterLuid(),IID_PPV_ARGS(adapter.Out())))||
               FAILED(adapter->GetDesc1(&desc)))throw std::runtime_error("DXGI adapter identification failed");
            char name[512]{};
            if(!WideCharToMultiByte(CP_UTF8,0,desc.Description,-1,name,sizeof name,nullptr,nullptr)||!name[0])
                throw std::runtime_error("DXGI adapter name missing");
            // DXVK appends the Vulkan driver/ISA label; HIP exposes the base
            // product name. Strip this known trailing decoration only.
            if(char* driver=std::strstr(name," (RADV ")){
                if(name[std::strlen(name)-1]==')')*driver=0;
            }
            auto client=std::make_shared<NativeHipClient>();client->Create(name);
            if(!client->Ready())throw std::runtime_error("HIP client not ready");
            s->client=std::move(client);
            Log("ready",0,name);Log("codec",0,s->display_srgb?"explicit display-sRGB; DLSS5_CODEC_SRGB=1":"linear mode=1 paper_white=1");
            Log("live_network",0,"enabled; F6 toggles bypass; reset every frame, NOT temporal; no performance claim");
            s->init.store(2,std::memory_order_release);
        }catch(const std::exception& e){Log("init_failed",0,e.what());s->init=-1;s->disabled=true;}
        catch(...){Log("init_failed");s->init=-1;s->disabled=true;}
    }
public:
    NativeHipLive()=default;
    NativeHipLive(const NativeHipLive&)=delete;
    NativeHipLive& operator=(const NativeHipLive&)=delete;
    // True means accepted marker AND suffix recorded, not GPU completion.
    // Owner destruction cannot invalidate background init or retained jobs.
    bool Record(ID3D12GraphicsCommandList* list,ID3D12Resource* color,D3D12_RESOURCE_STATES state,uint64_t frame_id) noexcept {
        auto s=state_;
        const bool down=(GetAsyncKeyState(VK_F6)&0x8000)!=0;
        const bool was=s->key_down.exchange(down);
        if(down&&!was){auto g=s->generation.fetch_add(1)+1;Log("F6",frame_id,(g&1)?"bypass; original pixels":"enabled; live network (not temporal)");}
        const uint64_t generation=s->generation.load();
        try {
            std::unique_lock<std::mutex> lock(s->recording,std::try_to_lock);
            if(!lock.owns_lock()||s->disabled||(generation&1)||!list||!color)return false;
            Ref<ID3D12GraphicsCommandList> hold_list(list);
            if(list->GetType()!=D3D12_COMMAND_LIST_TYPE_DIRECT)return false;
            Ref<ID3D12Device> device,color_device;
            if(FAILED(list->GetDevice(IID_PPV_ARGS(device.Out())))||
               FAILED(color->GetDevice(IID_PPV_ARGS(color_device.Out())))||
               !NativeSameDevice(device.p,color_device.p))return false;
            // Never unwrap the foreign resource. Only validate device identity.
            const auto desc=color->GetDesc();const auto format=NativePixelFormatFromDxgi(unsigned(desc.Format));
            if(desc.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D||desc.Width>7680||
               !NativePixelGeometry(unsigned(desc.Width),desc.Height)||desc.MipLevels!=1||
               desc.DepthOrArraySize!=1||desc.SampleDesc.Count!=1||desc.SampleDesc.Quality!=0||
               (desc.Flags&D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS)||
               !NativePixelBytes(format)||!ValidState(state))return false;
            if(s->device.p&&!NativeSameDevice(s->device.p,device.p))return false;
            if(s->init==0){
                // Pin this add-on's callback code as well as its heap state. No
                // FreeLibrary race on detached initialization / allocator refs.
                HMODULE module=nullptr;
                if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_PIN,
                    reinterpret_cast<LPCWSTR>(&Initialize),&module)){s->disabled=true;return false;}
                s->device.p=device.p;device->AddRef();
                const wchar_t* value=_wgetenv(L"DLSS5_CODEC_SRGB");s->display_srgb=value&&!wcscmp(value,L"1");
                s->init=1;Log("initializing",frame_id,"background HIP init; original frame unchanged");
                try{std::thread(&Initialize,s).detach();}catch(...){s->init=-1;s->disabled=true;throw;}
                return false;
            }
            if(s->init.load(std::memory_order_acquire)!=2)return false;
            if(s->inflight.fetch_add(1)>=MaxInflight){s->inflight.fetch_sub(1);return false;}
            Job* raw=nullptr;
            try{raw=new Job(s,frame_id,generation,format,unsigned(desc.Width),desc.Height);}catch(...){s->inflight.fetch_sub(1);throw;}
            struct Caller {Job* p;~Caller(){p->Release();}} caller{raw};
            if(!raw->resources.Create(device.p,color,desc,NativePixelBytes(format))){Log("setup_failed",frame_id,"original unchanged");return false;}
            if(FAILED(list->SetPrivateDataInterface(raw->anchor,raw))){Log("anchor_failed",frame_id);return false;}
            Copy(list,*raw,state,false);
            uint32_t accepted=0;
            Dlss5SubmitMarker marker{DLSS5_SUBMIT_MAGIC,DLSS5_SUBMIT_VERSION,sizeof(Dlss5SubmitMarker),raw,
                &Job::Retain,&Job::ReleaseContext,&Job::Run,&accepted};
            list->SetMarker(DLSS5_SUBMIT_METADATA,&marker,sizeof marker);
            if(accepted!=1){
                if(!s->disabled.exchange(true))Log("disabled",frame_id,"missing vkd3d HIP submit extension; no suffix writeback");
                return false;
            }
            Copy(list,*raw,state,true);
            // Extension now owns the job through allocator GPU completion/reset.
            // Failure to remove the conservative anchor is safe, not premature release.
            list->SetPrivateDataInterface(raw->anchor,nullptr);
            Log("queued",frame_id,"prefix -> HIP callback -> suffix");return true;
        }catch(const std::exception& e){Log("record_failed",frame_id,e.what());return false;}
        catch(...){Log("record_failed",frame_id);return false;}
    }
};
