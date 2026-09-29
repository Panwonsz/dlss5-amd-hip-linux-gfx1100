/* libdlss5_hip_preload.so: the object to LD_PRELOAD into a Proton game (DLSS5_NR_INPROC, 2026-09-29).

   It carries no HIP. Its constructor publishes the same three bridges as capi.hip (same magics, same
   environment variables), so dlss5_hip.dll finds it exactly as it found the old preload. The model
   library is loaded by the FIRST CALL through a bridge, with dlopen, in the thread that makes the call.

   Why: capi.hip records that inside Proton the process calling the model had the library mapped
   without its constructors having run there, and HIP then reported no device, because the fat binary
   carrying the kernels is registered by a constructor. Loading the real library with dlopen at call
   time runs its constructors in the process that uses it, whatever happened at exec time. (Upstream
   solved the same problem with a dlopen preload, 7e3a3cf.)

   The real library: DLSS5_HIP_LIBRARY if set, else libdlss5_hip.so beside this file. A failed dlopen
   is reported through dlss5_last_error with dlerror's text, which is what names a missing ROCm
   dependency inside Steam's container. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dlss5_capi.h"

typedef int (*init_fn)(const char *, int);
typedef int (*run_fn)(const float *, float *, dlss5_u32);
typedef void (*shutdown_fn)(void);
typedef const char *(*error_fn)(void);
typedef int (*frame_fn)(const Dlss5Frame *);
typedef int (*find_fn)(const char *);

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static void *g_lib;
static init_fn r_init;
static run_fn r_run;
static shutdown_fn r_shutdown;
static error_fn r_error;
static frame_fn r_frame;
static find_fn r_find;
static char g_error[1024];

static Dlss5HipBridge g_bridge;
static Dlss5HipFrameBridge g_frame_bridge;
static Dlss5HipDeviceBridge g_device_bridge;

static int load(void) {
    int ok;
    pthread_mutex_lock(&g_lock);
    if (!g_lib) {
        char beside[4096];
        const char *path = getenv("DLSS5_HIP_LIBRARY");
        if (!path || !path[0]) {
            Dl_info info;
            const char *slash;
            beside[0] = 0;
            if (dladdr((void *)&load, &info) && info.dli_fname && (slash = strrchr(info.dli_fname, '/')))
                snprintf(beside, sizeof beside, "%.*s/libdlss5_hip.so", (int)(slash - info.dli_fname),
                         info.dli_fname);
            else
                snprintf(beside, sizeof beside, "libdlss5_hip.so");
            path = beside;
        }
        void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!h) {
            const char *why = dlerror();
            snprintf(g_error, sizeof g_error, "preload shim: dlopen %s failed: %s", path, why ? why : "?");
        } else {
            r_init = (init_fn)dlsym(h, "dlss5_init");
            r_run = (run_fn)dlsym(h, "dlss5_run");
            r_shutdown = (shutdown_fn)dlsym(h, "dlss5_shutdown");
            r_error = (error_fn)dlsym(h, "dlss5_last_error");
            r_frame = (frame_fn)dlsym(h, "dlss5_run_frame");
            r_find = (find_fn)dlsym(h, "dlss5_find_device");
            if (!r_init || !r_run || !r_shutdown || !r_error)
                snprintf(g_error, sizeof g_error, "preload shim: %s lacks the dlss5_* exports", path);
            else
                g_lib = h; /* never closed: HIP does not survive being unloaded */
        }
    }
    ok = g_lib != NULL;
    pthread_mutex_unlock(&g_lock);
    return ok;
}

static int s_init(const char *weights, int gpu) { return load() ? r_init(weights, gpu) : -1; }
static int s_run(const float *in, float *out, dlss5_u32 seed) { return load() ? r_run(in, out, seed) : -1; }
static void s_shutdown(void) {
    if (g_lib) /* never load the model just to shut it down */
        r_shutdown();
}
static const char *s_error(void) {
    if (g_lib)
        return r_error();
    return g_error[0] ? g_error : "preload shim: model library not loaded yet";
}
static int s_frame(const Dlss5Frame *f) {
    if (!load())
        return -1;
    if (!r_frame) {
        snprintf(g_error, sizeof g_error, "preload shim: model library lacks dlss5_run_frame");
        return -1;
    }
    return r_frame(f);
}
static int s_find(const char *name) {
    if (!load())
        return -1;
    if (!r_find) {
        snprintf(g_error, sizeof g_error, "preload shim: model library lacks dlss5_find_device");
        return -1;
    }
    return r_find(name);
}

static void publish(const char *name, void *table, unsigned long long *magic) {
    char buf[32];
    snprintf(buf, sizeof buf, "%p", table);
    if (setenv(name, buf, 1) != 0) {
        *magic = 0;
        (void)unsetenv(name);
    }
}

__attribute__((constructor)) static void dlss5_preload_publish(void) {
    g_bridge.magic = DLSS5_HIP_MAGIC;
    g_bridge.init = s_init;
    g_bridge.run = s_run;
    g_bridge.shutdown = s_shutdown;
    g_bridge.last_error = s_error;
    publish(DLSS5_HIP_ENV, &g_bridge, &g_bridge.magic);
    g_frame_bridge.magic = DLSS5_HIP_FRAME_MAGIC;
    g_frame_bridge.run_frame = s_frame;
    publish(DLSS5_HIP_FRAME_ENV, &g_frame_bridge, &g_frame_bridge.magic);
    g_device_bridge.magic = DLSS5_HIP_DEVICE_MAGIC;
    g_device_bridge.find_device = s_find;
    publish(DLSS5_HIP_DEVICE_ENV, &g_device_bridge, &g_device_bridge.magic);
}
