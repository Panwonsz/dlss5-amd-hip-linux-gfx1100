#include "dlss5_capi.h"
#include "dlss5_network.hpp"
#include "dlss5_launch.hpp"
#include "dlss5_frame.hpp"
#include "dlss5_staged_io.hpp"
#include "dlss5_shared_import.hpp"
#include <chrono>
#include <mutex>
#include <string>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <memory>
#include <unistd.h>

static std::mutex g_mu;
static dlss5::Network70* g_net;
static thread_local std::string g_err;
static Dlss5HipBridge g_bridge;
static Dlss5HipFrameBridge g_frame_bridge;
static Dlss5HipDeviceBridge g_device_bridge;
static int g_gpu;
struct FrameBuffers {
    dlss5::DeviceBuf source,proxy,neural,decoded,history,motion,warped,base;
    std::vector<float> host_proxy,host_neural,host_warped,host_result;
    bool history_valid=false;
    dlss5_u32 previous_flags=0;
    float previous_white=0;
    void create() {
        const size_t n=1920ull*1080, padded=1920ull*1152;
        source.alloc(n*16);proxy.alloc(n*16);neural.alloc(n*12);decoded.alloc(n*12);history.alloc(n*12);
        warped.alloc(padded*16);base.alloc(padded*16);
        host_proxy.resize(n*4);host_neural.resize(padded*3);host_warped.resize(padded*4);host_result.resize(n*3);
    }
};
static std::unique_ptr<FrameBuffers> g_frame;
// dlss5_run_staged's device buffers (see the staged entry below).
struct StagedBuffers {
    dlss5::DeviceBuf in, out, rgba, rgb, flag;
    size_t bytes = 0;
};
static std::unique_ptr<StagedBuffers> g_staged;
static Dlss5HipStagedBridge g_staged_bridge;
static Dlss5HipSharedBridge g_shared_bridge;
// The network writes 1152 rows; the V1 caller wants the first 1080. Held here rather than allocated
// per call, and released by dlss5_shutdown so an unload does not leave 25 MB behind.
static std::vector<float> g_out1152;

/* Did this library's own initialisation run in THIS process?

   Inside a Proton game the answer turns out to matter more than anything about the sandbox. Wine's
   preloader reserves its address space and re-execs, and the process that finally calls us has our
   library mapped without ever having run its constructors -- the mappings came from an ancestor. That
   is invisible from outside and explains a string of dead ends: a preloaded probe whose constructor
   never fires, a bridge pointer that is valid because it was inherited rather than published here,
   and -- if it holds -- HIP reporting no device while ROCr lists a gfx1100 agent, because the fat
   binary carrying the kernels is registered by a constructor that never ran. */
static int g_ctor_pid;
__attribute__((constructor)) static void note_constructor() { g_ctor_pid = (int)getpid(); }

static void run_log(const char* s) {
    FILE* f = fopen("/tmp/dlss5-hip-run.log", "a");
    if (!f)
        return;
    fprintf(f, "%s\n", s);
    fclose(f);
}

/* Every frame, dlss5_run validated 15M floats one branch at a time, allocated and zeroed a 25 MB
   output vector, and copied the result out a row at a time. On the Ryzen 5 5600 that was 10-12 ms per
   answer -- about 8% of the in-game latency, and unlike model time it hands nothing back to the game,
   so all of it converts into a fresher image rather than into frame rate.

   The check itself stays. A float is non-finite exactly when its exponent field is all ones, which is
   the same predicate std::isfinite tests; accumulating it with OR instead of branching on it lets the
   compiler vectorise the scan. The one behavioural difference is that it no longer stops at the first
   bad value, which costs nothing on valid data and makes the timing deterministic. */
static bool all_finite(const float* p, size_t n) {
    uint32_t bad = 0;
    for (size_t i = 0; i < n; ++i) {
        uint32_t u;
        ::memcpy(&u, p + i, 4);
        bad |= uint32_t((u & 0x7F800000u) == 0x7F800000u);
    }
    return bad == 0;
}

static const char* set_err(const char* s) {
    g_err = s ? s : "";
    return g_err.c_str();
}

/* The HSA entry points this report uses, declared rather than included: the ROCr headers are not on
   the include path for this build, and only these three are wanted. */
extern "C" {
typedef struct { uint64_t handle; } dlss5_hsa_agent;
int hsa_init(void);
int hsa_iterate_agents(int (*callback)(dlss5_hsa_agent, void*), void* data);
int hsa_agent_get_info(dlss5_hsa_agent agent, int attribute, void* value);
}

/* What the runtime says about itself, appended to any init failure.
   The error string is the one channel that reliably leaves this library: it is returned through the
   trampoline to the caller, which logs it. Files written from here land in whatever /tmp the process
   ended up with -- inside Steam's container that is a private one -- and preloaded probes have their
   constructors run in an ancestor rather than in the process that actually calls us. So the report
   travels with the error instead of being left somewhere to be found. */
struct HsaTally { int agents = 0, gpus = 0; char first[64] = "-"; };

static int tally_agent(dlss5_hsa_agent agent, void* data) {
    auto* t = static_cast<HsaTally*>(data);
    char name[64] = {0};
    int kind = -1;
    hsa_agent_get_info(agent, 0 /* NAME */, name);
    hsa_agent_get_info(agent, 17 /* DEVICE */, &kind);
    t->agents++;
    if (kind == 1) { /* GPU */
        if (t->gpus == 0) snprintf(t->first, sizeof t->first, "%s", name);
        t->gpus++;
    }
    return 0; /* HSA_STATUS_SUCCESS */
}

static std::string device_report() {
    int count = -1, runtime = 0, driver = 0;
    const hipError_t counted = hipGetDeviceCount(&count);
    hipRuntimeGetVersion(&runtime);
    hipDriverGetVersion(&driver);
    char name[128] = "-";
    if (count > 0) {
        hipDeviceProp_t props{};
        if (hipGetDeviceProperties(&props, 0) == hipSuccess)
            snprintf(name, sizeof name, "%s/%s", props.name, props.gcnArchName);
    }
    /* The layer below HIP, asked separately. HIP reporting no devices while ROCr lists a GPU agent
       means HIP is rejecting a card the driver can see; ROCr listing none, in a process where the
       same library lists one outside the container, means ROCr itself cannot come up here -- and
       that is the difference between a fixable configuration and a design change. */
    HsaTally hsa;
    const int hsa_status = hsa_init();
    if (hsa_status == 0)
        hsa_iterate_agents(tally_agent, &hsa);

    char buf[768];
    snprintf(buf, sizeof buf,
             " [pid=%d devices=%d (%s) first=%s runtime=%d driver=%d"
             " | hsa_init=%d agents=%d gpus=%d first_agent=%s | ctor_pid=%d]",
             (int)getpid(), count, hipGetErrorString(counted), name, runtime, driver,
             hsa_status, hsa.agents, hsa.gpus, hsa.first, g_ctor_pid);
    return buf;
}

extern "C" int dlss5_init(const char* weights_dir, int gpu) {
    std::lock_guard<std::mutex> lock(g_mu);
    try {
        const char* dir = weights_dir && weights_dir[0] ? weights_dir : getenv("DLSS5_HIP_WEIGHTS");
        if (!dir || !dir[0])
            return set_err("no weights dir (DLSS5_HIP_WEIGHTS)"), -1;
        const int candidate_gpu = gpu >= 0 ? gpu : 0;
        dlss5::hip_check(hipSetDevice(candidate_gpu), "hipSetDevice");
        auto candidate = std::make_unique<dlss5::Network70>();
        candidate->create(dir);
        if (g_net) dlss5::hip_check(hipSetDevice(g_gpu), "old device");
        delete g_net;
        g_frame.reset();
        g_net = candidate.release();
        g_gpu = candidate_gpu;
        g_err.clear();
        run_log("init ok");
        return 0;
    } catch (const std::exception& e) {
        const std::string detail = std::string(e.what()) + device_report();
        run_log(detail.c_str());
        set_err(detail.c_str());
        return -1;
    }
}

extern "C" int dlss5_run(const float* rgba1080, float* rgb1080, dlss5_u32 seed) {
    std::lock_guard<std::mutex> lock(g_mu);
    // V1 has no motion contract. Even a rejected call breaks frame continuity.
    if (g_frame) g_frame->history_valid=false;
    if (!g_net)
        return set_err("not initialized"), -1;
    if (!rgba1080 || !rgb1080 || rgba1080 == rgb1080)
        return set_err("null buffer"), -1;
    try {
        if (!all_finite(rgba1080, 1920ull * 1080 * 4))
            throw std::runtime_error("nonfinite network input");
        dlss5::hip_check(hipSetDevice(g_gpu), "hipSetDevice");
        // Held across calls rather than allocated per frame: 25 MB of fresh pages, zeroed by the
        // vector's value-initialisation and then immediately overwritten by the network. g_mu
        // serialises every caller, so one buffer is enough.
        if (g_out1152.size() != 1920ull * 1152 * 3) g_out1152.resize(1920ull * 1152 * 3);
        g_net->run(rgba1080, g_out1152.data(), seed);
        if (!all_finite(g_out1152.data(), g_out1152.size()))
            throw std::runtime_error("nonfinite network output");
        // Source and destination advance by the same 1920*3 stride, so the 1080 rows the caller
        // wants are one contiguous block at the front of the 1152-row buffer -- one memcpy, not 1080.
        std::memcpy(rgb1080, g_out1152.data(), 1920ull * 1080 * 3 * 4);
        static int nlog;
        if (nlog < 4) {
            float md = 0.f, acc = 0.f;
            for (unsigned i = 0; i < 1920u * 1080u; i++) {
                float d0 = rgb1080[i * 3] - rgba1080[i * 4];
                float d1 = rgb1080[i * 3 + 1] - rgba1080[i * 4 + 1];
                float d2 = rgb1080[i * 3 + 2] - rgba1080[i * 4 + 2];
                float a = fabsf(d0) + fabsf(d1) + fabsf(d2);
                if (a > md)
                    md = a;
                acc += a;
            }
            char buf[128];
            snprintf(buf, sizeof buf, "run ok  gpu=%.2fms  max|d|=%.4f  mean|d|=%.4f  rgb0=%.3f %.3f %.3f",
                     g_net->last_gpu_ms(), md, acc / float(1920u * 1080u), rgb1080[0], rgb1080[1],
                     rgb1080[2]);
            run_log(buf);
            nlog++;
        }
        g_err.clear();
        return 0;
    } catch (const std::exception& e) {
        set_err(e.what());
        run_log(e.what());
        return -1;
    }
}

extern "C" void dlss5_shutdown(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_net && hipSetDevice(g_gpu) != hipSuccess) {
        set_err("cannot select model device for shutdown");
        return;
    }
    delete g_net;
    g_net = nullptr;
    g_frame.reset();
    g_staged.reset();
    std::vector<float>().swap(g_out1152);
}

extern "C" const char* dlss5_last_error(void) { return g_err.c_str(); }

extern "C" double dlss5_last_gpu_ms(void) {
    // The GPU interval of the last dlss5_run, as Network70's own events measured it. Subtracting it
    // from the wall time of the call separates the model from this wrapper's host-side work --
    // validation, the output vector, the copy -- which no GPU-side benchmark has ever covered.
    std::lock_guard<std::mutex> lock(g_mu);
    return g_net ? double(g_net->last_gpu_ms()) : 0.0;
}

extern "C" const char* dlss5_formats(void) {
    // dlss5_w16() reads the environment once, on first call, and dlss5_kernel_formats() is compiled
    // into kernels.o -- so this reports the weight format this process will actually use and the
    // activation formats its kernels were built with, not what the caller's environment says now.
    static const std::string s = std::string("weights=") + (dlss5::dlss5_w16() ? "f16" : "e4m3")
                               + " kernels[" + dlss5::dlss5_kernel_formats() + "]";
    return s.c_str();
}

extern "C" int dlss5_find_device(const char* name) {
    std::lock_guard<std::mutex> lock(g_mu);
    try {
        if (!name || !*name) throw std::runtime_error("graphics adapter name required");
        int count=0, selected=-1;
        dlss5::hip_check(hipGetDeviceCount(&count), "HIP enumeration");
        for (int i=0;i<count;++i) {
            char candidate[256]{};
            dlss5::hip_check(hipDeviceGetName(candidate,sizeof candidate,i), "HIP adapter name");
            if (!std::strcmp(name,candidate)) {
                if (selected!=-1) throw std::runtime_error("ambiguous graphics/HIP adapter name");
                selected=i;
            }
        }
        if (selected<0) throw std::runtime_error("graphics adapter not found in HIP");
        g_err.clear();
        return selected;
    } catch(const std::exception& e) {set_err(e.what());return -1;}
}

extern "C" int dlss5_run_frame(const Dlss5Frame* f) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_net) return set_err("not initialized"), -1;
    if (!f || f->struct_size != sizeof(Dlss5Frame) || (f->flags & ~1u) ||
        !f->rgba || !f->rgb || f->rgb == f->rgba || f->reset > 1) {
        if (g_frame) g_frame->history_valid=false;
        return set_err("invalid frame ABI/buffers"), -1;
    }
    try {
        const size_t n=1920ull*1080;
        for (size_t i=0;i<n*4;++i) if (!std::isfinite(f->rgba[i])) throw std::runtime_error("nonfinite frame input");
        if (!std::isfinite(f->motion_uv_scale_x)||!std::isfinite(f->motion_uv_scale_y)) throw std::runtime_error("nonfinite motion scale");
        if (f->motion) {
            if (!f->motion_width||!f->motion_height||f->motion_width>7680||f->motion_height>4320) throw std::runtime_error("motion extent");
            for (size_t i=0;i<size_t(f->motion_width)*f->motion_height*2;++i)
                if (!std::isfinite(f->motion[i])) throw std::runtime_error("nonfinite motion");
        }
        dlss5::hip_check(hipSetDevice(g_gpu),"frame device");
        if (!g_frame) {auto next=std::make_unique<FrameBuffers>();next->create();g_frame=std::move(next);}
        auto& b=*g_frame;
        if (f->reset || f->flags!=b.previous_flags || f->paper_white!=b.previous_white) b.history_valid=false;
        b.previous_flags=f->flags;b.previous_white=f->paper_white;
        dlss5::FrameSettings settings{f->paper_white,f->transfer,f->color,(f->flags&1u)!=0};
        b.source.upload(f->rgba,n*16);
        dlss5::frame_encode(b.source.as<float>(),b.proxy.as<float>(),1920,1080,settings,nullptr);
        dlss5::hip_check(hipStreamSynchronize(nullptr),"frame encode");b.proxy.download(b.host_proxy.data(),n*16);
        bool temporal=b.history_valid&&f->motion;
        if (temporal) {
            size_t bytes=size_t(f->motion_width)*f->motion_height*8;
            if (b.motion.bytes!=bytes)b.motion.alloc(bytes);
            b.motion.upload(f->motion,bytes);
            dlss5::frame_warp(b.proxy.as<float>(),b.history.as<float>(),b.motion.as<float>(),b.warped.as<float>(),
                1920,1080,1920,1152,f->motion_width,f->motion_height,f->motion_uv_scale_x,f->motion_uv_scale_y,false,nullptr);
            dlss5::frame_warp(b.proxy.as<float>(),nullptr,nullptr,b.base.as<float>(),1920,1080,1920,1152,0,0,0,0,true,nullptr);
            dlss5::frame_guard(b.base.as<float>(),b.warped.as<float>(),1920*1152,3.f/255,12.f/255,nullptr);
            dlss5::hip_check(hipStreamSynchronize(nullptr),"frame history");b.warped.download(b.host_warped.data(),b.host_warped.size()*4);
        }
        g_net->run(b.host_proxy.data(),b.host_neural.data(),f->seed,temporal?b.host_warped.data():nullptr);
        for (float v:b.host_neural) if (!std::isfinite(v)) throw std::runtime_error("nonfinite neural output");
        b.neural.upload(b.host_neural.data(),n*12);
        if (temporal) dlss5::frame_smooth(b.warped.as<float>(),b.neural.as<float>(),unsigned(n),10.f/255,.8f,nullptr);
        dlss5::frame_decode(b.source.as<float>(),b.proxy.as<float>(),b.neural.as<float>(),b.decoded.as<float>(),1920,1080,settings,nullptr);
        dlss5::hip_check(hipStreamSynchronize(nullptr),"frame decode");b.decoded.download(b.host_result.data(),n*12);
        for (float v:b.host_result) if (!std::isfinite(v)) throw std::runtime_error("nonfinite decoded output");
        dlss5::hip_check(hipMemcpy(b.history.p,b.neural.p,n*12,hipMemcpyDeviceToDevice),"save history");b.history_valid=true;
        std::memcpy(f->rgb,b.host_result.data(),n*12);g_err.clear();return 0;
    } catch(const std::exception&e) {if(g_frame)g_frame->history_valid=false;set_err(e.what());return -1;}
}

/* ---- Staged entry (2026-09-29): the daemon's whole exchange, in this process, on the GPU. ----
   in/out are the game's CPU-mapped D3D12 READBACK / UPLOAD buffers (or any host memory): 1920x1080 in
   DXGI `format`, rows `pitch` bytes apart. out receives exactly the bytes dlss5-nr-daemon sends back for
   the same request and seed: in's bytes (row padding included) with every pixel replaced by
   EncodeRow(dlss5_run(DecodeRow(in))). Non-finite model input or output fails the call, as dlss5_run
   does. hip/tests/test_staged.cpp checks the byte equality against the daemon's own code. */

// The part dlss5_run_staged and dlss5_run_shared have in common: dev_in / dev_out hold the staged
// layout (1920x1080 in `format`, rows `pitch` apart) in device memory. Decode, model, the finite checks
// dlss5_run makes, in -> out (the row padding the daemon echoes), encode. Synchronous.
static void staged_core(StagedBuffers& s, const uint8_t* dev_in, uint8_t* dev_out, dlss5_u32 format,
                        dlss5_u32 pitch, dlss5_u32 seed) {
    const size_t bytes = size_t(pitch) * 1080;
    const dim3 grid((1920 + 255) / 256, 1080), block(256);
    uint32_t bad = 0;
    dlss5::hip_check(hipMemset(s.flag.p, 0, 4), "staged flag");
    hipLaunchKernelGGL(dlss5::staged::k_staged_decode, grid, block, 0, nullptr, dev_in, s.rgba.as<float4>(),
                       format, pitch, 1920u, s.flag.as<uint32_t>());
    dlss5::hip_check(hipGetLastError(), "staged decode");
    dlss5::hip_check(hipMemcpy(&bad, s.flag.p, 4, hipMemcpyDeviceToHost), "staged decode check");
    if (bad) throw std::runtime_error("nonfinite network input");
    g_net->run(s.rgba.as<float>(), s.rgb.as<float>(), seed); // device in, device out
    dlss5::hip_check(hipMemset(s.flag.p, 0, 4), "staged flag");
    hipLaunchKernelGGL(dlss5::staged::k_staged_finite, dim3(1024), dim3(256), 0, nullptr, s.rgb.as<float>(),
                       size_t(1920) * 1152 * 3, s.flag.as<uint32_t>());
    dlss5::hip_check(hipGetLastError(), "staged output check");
    dlss5::hip_check(hipMemcpy(&bad, s.flag.p, 4, hipMemcpyDeviceToHost), "staged output check");
    if (bad) throw std::runtime_error("nonfinite network output");
    dlss5::hip_check(hipMemcpy(dev_out, dev_in, bytes, hipMemcpyDeviceToDevice), "staged padding");
    hipLaunchKernelGGL(dlss5::staged::k_staged_encode, grid, block, 0, nullptr, s.rgb.as<float>(), dev_out, format,
                       pitch, 1920u);
    dlss5::hip_check(hipGetLastError(), "staged encode");
    dlss5::hip_check(hipDeviceSynchronize(), "staged completion");
}

static StagedBuffers& staged_buffers() {
    if (!g_staged) {
        auto next = std::make_unique<StagedBuffers>();
        next->rgba.alloc(1920ull * 1080 * 16);
        next->rgb.alloc(1920ull * 1152 * 12);
        next->flag.alloc(4);
        g_staged = std::move(next);
    }
    return *g_staged;
}

static bool staged_args_ok(dlss5_u32 format, dlss5_u32 pitch) {
    const dlss5_u32 bpp = dlss5::staged::bytes_per_pixel(format);
    return bpp && pitch >= 1920u * bpp && pitch % 4 == 0 && pitch <= (1u << 20);
}

extern "C" int dlss5_run_staged(const void* in, void* out, dlss5_u32 format, dlss5_u32 pitch,
                                dlss5_u32 seed) {
    std::lock_guard<std::mutex> lock(g_mu);
    // Like dlss5_run: no motion contract, so any call breaks frame continuity.
    if (g_frame) g_frame->history_valid = false;
    if (!g_net) return set_err("not initialized"), -1;
    if (!in || !out || in == out || !staged_args_ok(format, pitch))
        return set_err("staged: unsupported format, bad pitch or null buffers"), -1;
    try {
        dlss5::hip_check(hipSetDevice(g_gpu), "hipSetDevice");
        const size_t bytes = size_t(pitch) * 1080;
        auto& s = staged_buffers();
        if (s.bytes != bytes) {
            s.in.alloc(bytes);
            s.out.alloc(bytes);
            s.bytes = bytes;
        }
        dlss5::hip_check(hipMemcpy(s.in.p, in, bytes, hipMemcpyHostToDevice), "staged upload");
        staged_core(s, s.in.as<uint8_t>(), s.out.as<uint8_t>(), format, pitch, seed);
        dlss5::hip_check(hipMemcpy(out, s.out.p, bytes, hipMemcpyDeviceToHost), "staged download");
        g_err.clear();
        return 0;
    } catch (const std::exception& e) {
        set_err(e.what());
        run_log(e.what());
        return -1;
    }
}

/* ---- Zero-copy (2026-09-29): the same exchange on the game's own GPU buffers. ----
   in_handle / out_handle: Win32 shared handles (ID3D12Device::CreateSharedHandle) of DEFAULT-heap
   buffers of `buffer_bytes`, created with D3D12_HEAP_FLAG_SHARED; `generation` changes whenever the caller
   recreates them. They are imported once (see dlss5_shared_import.hpp) and then read and written in
   place: no host copy at all. out receives exactly what dlss5_run_staged would write.
   verify_in, if not null, is a host copy of the same frame's input (the caller staged it twice); the
   imported input is compared with it byte for byte, and a difference returns 1 (the answer is still
   written) with the count in dlss5_last_error. */
extern "C" int dlss5_import_shared(void* handle, unsigned long long buffer_bytes, dlss5_u32 generation) {
    std::lock_guard<std::mutex> lock(g_mu);
    try {
        dlss5::hip_check(hipSetDevice(g_gpu), "hipSetDevice");
        void* dev = nullptr;
        std::string why;
        if (!dlss5::shared::import_handle(handle, size_t(buffer_bytes), generation, &dev, why))
            return set_err(("zero-copy import: " + why).c_str()), -1;
        g_err.clear();
        return 0;
    } catch (const std::exception& e) {
        set_err(e.what());
        return -1;
    }
}

extern "C" int dlss5_run_shared(void* in_handle, void* out_handle, unsigned long long buffer_bytes,
                                dlss5_u32 generation, dlss5_u32 format, dlss5_u32 pitch, dlss5_u32 seed,
                                const void* verify_in) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_frame) g_frame->history_valid = false;
    if (!g_net) return set_err("not initialized"), -1;
    const size_t bytes = size_t(pitch) * 1080;
    if (!in_handle || !out_handle || in_handle == out_handle || !staged_args_ok(format, pitch) ||
        buffer_bytes < bytes)
        return set_err("shared: unsupported format, bad pitch/size or null handles"), -1;
    try {
        dlss5::hip_check(hipSetDevice(g_gpu), "hipSetDevice");
        void *dev_in = nullptr, *dev_out = nullptr;
        std::string why;
        if (!dlss5::shared::import_handle(in_handle, size_t(buffer_bytes), generation, &dev_in, why) ||
            !dlss5::shared::import_handle(out_handle, size_t(buffer_bytes), generation, &dev_out, why))
            return set_err(("zero-copy import: " + why).c_str()), -1;
        size_t differing = 0;
        if (verify_in) {
            static std::vector<uint8_t> seen;
            seen.resize(bytes);
            dlss5::hip_check(hipMemcpy(seen.data(), dev_in, bytes, hipMemcpyDeviceToHost), "verify download");
            const auto* host = static_cast<const uint8_t*>(verify_in);
            // Pixel bytes only: the copies never write the row padding, so it holds whatever was there.
            const size_t row_bytes = size_t(1920) * dlss5::staged::bytes_per_pixel(format);
            for (size_t y = 0; y < 1080; ++y)
                for (size_t x = 0; x < row_bytes; ++x) differing += seen[y * pitch + x] != host[y * pitch + x];
        }
        staged_core(staged_buffers(), static_cast<const uint8_t*>(dev_in), static_cast<uint8_t*>(dev_out), format,
                    pitch, seed);
        if (differing) {
            char buf[160];
            snprintf(buf, sizeof buf, "zero-copy verify: the shared input differs from the mapped copy in %zu of %zu pixel bytes",
                     differing, size_t(1920) * dlss5::staged::bytes_per_pixel(format) * 1080);
            set_err(buf);
            return 1;
        }
        g_err.clear();
        return 0;
    } catch (const std::exception& e) {
        set_err(e.what());
        run_log(e.what());
        return -1;
    }
}

__attribute__((constructor)) static void dlss5_hip_publish(void) {
    memset(&g_bridge, 0, sizeof(g_bridge));
    g_bridge.magic = DLSS5_HIP_MAGIC;
    g_bridge.init = dlss5_init;
    g_bridge.run = dlss5_run;
    g_bridge.shutdown = dlss5_shutdown;
    g_bridge.last_error = dlss5_last_error;
    char buf[32];
    snprintf(buf, sizeof(buf), "%p", (void*)&g_bridge);
    if (setenv(DLSS5_HIP_ENV, buf, 1) != 0) {
        g_bridge.magic = 0;
        (void)unsetenv(DLSS5_HIP_ENV);
    }
    g_frame_bridge.magic=DLSS5_HIP_FRAME_MAGIC;g_frame_bridge.run_frame=dlss5_run_frame;
    snprintf(buf,sizeof(buf),"%p",(void*)&g_frame_bridge);
    if(setenv(DLSS5_HIP_FRAME_ENV,buf,1)!=0){g_frame_bridge.magic=0;(void)unsetenv(DLSS5_HIP_FRAME_ENV);}
    g_device_bridge.magic=DLSS5_HIP_DEVICE_MAGIC;g_device_bridge.find_device=dlss5_find_device;
    snprintf(buf,sizeof(buf),"%p",(void*)&g_device_bridge);
    if(setenv(DLSS5_HIP_DEVICE_ENV,buf,1)!=0){g_device_bridge.magic=0;(void)unsetenv(DLSS5_HIP_DEVICE_ENV);}
    g_staged_bridge.magic=DLSS5_HIP_STAGED_MAGIC;g_staged_bridge.run_staged=dlss5_run_staged;g_staged_bridge.last_gpu_ms=dlss5_last_gpu_ms;
    snprintf(buf,sizeof(buf),"%p",(void*)&g_staged_bridge);
    if(setenv(DLSS5_HIP_STAGED_ENV,buf,1)!=0){g_staged_bridge.magic=0;(void)unsetenv(DLSS5_HIP_STAGED_ENV);}
    g_shared_bridge.magic=DLSS5_HIP_SHARED_MAGIC;g_shared_bridge.import_shared=dlss5_import_shared;g_shared_bridge.run_shared=dlss5_run_shared;
    snprintf(buf,sizeof(buf),"%p",(void*)&g_shared_bridge);
    if(setenv(DLSS5_HIP_SHARED_ENV,buf,1)!=0){g_shared_bridge.magic=0;(void)unsetenv(DLSS5_HIP_SHARED_ENV);}
}
