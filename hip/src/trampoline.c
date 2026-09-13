/* PE trampoline: Win64 exports -> Linux table in this Wine process only. */
#define DLSS5_CAPI_EXPORT __declspec(dllexport)
#include "../include/dlss5_capi.h"

#define EXPORT __declspec(dllexport)
#define SYSV __attribute__((sysv_abi))

typedef int BOOL;
typedef unsigned long DWORD;
typedef void *HANDLE;
#define DLL_PROCESS_ATTACH 1

__declspec(dllimport) HANDLE __stdcall GetModuleHandleA(const char *n);
__declspec(dllimport) void *__stdcall GetProcAddress(HANDLE m, const char *n);

static Dlss5HipBridge *g;
static Dlss5HipFrameBridge *g_frame;
static Dlss5HipDeviceBridge *g_device;
static _Thread_local const char *local_error;

static unsigned long long parse_hex_ptr(const char *s) {
    unsigned long long v = 0;
    unsigned digits = 0;
    if (!s)
        return 0;
    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
        s += 2;
    while (*s) {
        unsigned digit;
        unsigned char c = (unsigned char)*s++;
        if (c >= '0' && c <= '9')
            digit = (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f')
            digit = (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F')
            digit = (unsigned)(c - 'A' + 10);
        else
            return 0;
        if (++digits > 16)
            return 0;
        v = (v << 4) | digit;
    }
    return digits ? v : 0;
}

static int bridge_ok(void) { return g && g->magic == DLSS5_HIP_MAGIC; }

static void *read_bridge(const char *name) {
    char buf[64] = {0};
    unsigned i;
    typedef int (__stdcall *wine_env_fn)(const char *, char *, unsigned long long);
    HANDLE ntdll = GetModuleHandleA("ntdll.dll");
    wine_env_fn unix_env = ntdll ? (wine_env_fn)GetProcAddress(ntdll, "__wine_get_unix_env") : 0;

    /* Only the live Unix environment sees the preload constructor's setenv.
       Never use Wine's inherited PE snapshot or any filesystem pointer record.
       The trusted preload must overwrite this value in EVERY exec'd process
       and keep the table/library resident. This is not an untrusted IPC API. */
    if (!unix_env || unix_env(name, buf, sizeof(buf)) != 0)
        return 0;
    for (i = 0; i < sizeof(buf) && buf[i]; ++i) {}
    if (i == sizeof(buf))
        return 0;
    return (void *)parse_hex_ptr(buf);
}

static void init_bridge(void) {
    g = (Dlss5HipBridge *)read_bridge(DLSS5_HIP_ENV);
    g_frame = (Dlss5HipFrameBridge *)read_bridge(DLSS5_HIP_FRAME_ENV);
    g_device = (Dlss5HipDeviceBridge *)read_bridge(DLSS5_HIP_DEVICE_ENV);
}

BOOL __stdcall DllMain(void *h, DWORD reason, void *r) {
    (void)h;
    (void)r;
    if (reason == DLL_PROCESS_ATTACH)
        init_bridge();
    return 1;
}

EXPORT int dlss5_init(const char *weights_dir, int gpu) {
    local_error = 0;
    if (!bridge_ok() || !g->init) {
        local_error = "hip init bridge missing (live Wine Unix env + LD_PRELOAD required)";
        return -1;
    }
    return ((int SYSV (*)(const char *, int))g->init)(weights_dir, gpu);
}

EXPORT int dlss5_run(const float *rgba1080, float *rgb1080, unsigned seed) {
    local_error = 0;
    if (!bridge_ok() || !g->run) {
        local_error = "hip run bridge missing (live Wine Unix env + LD_PRELOAD required)";
        return -1;
    }
    return ((int SYSV (*)(const float *, float *, unsigned))g->run)(rgba1080, rgb1080, seed);
}

EXPORT void dlss5_shutdown(void) {
    local_error = 0;
    if (!bridge_ok() || !g->shutdown) {
        local_error = "hip shutdown bridge missing";
        return;
    }
    ((void SYSV (*)(void))g->shutdown)();
}

EXPORT const char *dlss5_last_error(void) {
    if (local_error)
        return local_error;
    if (!bridge_ok() || !g->last_error)
        return "hip bridge missing (live Wine Unix env + LD_PRELOAD libdlss5_hip.so required)";
    return ((const char *SYSV (*)(void))g->last_error)();
}

EXPORT int dlss5_run_frame(const Dlss5Frame *frame) {
    local_error = 0;
    if (!g_frame || g_frame->magic != DLSS5_HIP_FRAME_MAGIC || !g_frame->run_frame) {
        local_error = "hip frame bridge missing (rebuild and preload the matching V2 library)";
        return -1;
    }
    return ((int SYSV (*)(const Dlss5Frame *))g_frame->run_frame)(frame);
}

EXPORT int dlss5_find_device(const char *name) {
    local_error = 0;
    if (!g_device || g_device->magic != DLSS5_HIP_DEVICE_MAGIC || !g_device->find_device) {
        local_error = "hip device bridge missing (rebuild matching library)";
        return -1;
    }
    return ((int SYSV (*)(const char *))g_device->find_device)(name);
}
