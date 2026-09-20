/* C ABI published by libdlss5_hip.so and forwarded by the PE trampoline dlss5_hip.dll.
   All calls are synchronous. One process-global model/history is serialized
   internally; callers must coordinate ownership, shutdown and library residency.
   Host arrays must be full-sized, packed and non-overlapping. Output is untouched
   on failure. Every failed run/frame call invalidates temporal history; every V1
   run invalidates it, even on success. Successful init/shutdown resets it.
   Failed init preserves the previous model. last_error is thread-local and its
   returned pointer stays valid only until the next API call on that thread. */
#ifndef DLSS5_CAPI_H
#define DLSS5_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

#define DLSS5_HIP_MAGIC 0x3150494853534C44ULL /* "DLSSHIP1" */
#define DLSS5_HIP_ENV "DLSS5_HIP_BRIDGE"
/* Same-process transport only, read through Wine's live Unix environment. */

typedef unsigned int dlss5_u32;

typedef struct Dlss5HipBridge {
    unsigned long long magic;
    int (*init)(const char *weights_dir, int gpu);
    int (*run)(const float *rgba1080, float *rgb1080, dlss5_u32 seed);
    void (*shutdown)(void);
    const char *(*last_error)(void);
} Dlss5HipBridge;

#ifndef DLSS5_CAPI_EXPORT
#define DLSS5_CAPI_EXPORT
#endif
DLSS5_CAPI_EXPORT int dlss5_init(const char *weights_dir, int gpu);
DLSS5_CAPI_EXPORT int dlss5_run(const float *rgba1080, float *rgb1080, dlss5_u32 seed);
DLSS5_CAPI_EXPORT void dlss5_shutdown(void);
DLSS5_CAPI_EXPORT const char *dlss5_last_error(void);

/* The formats THIS library was built and configured with, e.g.
   "weights=f16 kernels[ah16=1 qkv16=0]". For logs and bug reports: the daemon
   outlives the game and is reused by the next launch, so an environment
   variable set for a run is not evidence about the process actually answering.
   Callers resolve it with dlsym and tolerate its absence in older builds. */
DLSS5_CAPI_EXPORT const char *dlss5_formats(void);

/* GPU milliseconds of the last dlss5_run. Wall time minus this is host-side
   work inside the library. Resolved with dlsym; absent in older builds. */
DLSS5_CAPI_EXPORT double dlss5_last_gpu_ms(void);

/* V2 uses a separate table, preserving the V1 ABI. Host pointers are packed
   1080p RGBA/RGB and a full motion rectangle (XY float32). flags bit0 selects
   display-sRGB; otherwise mode1 linear game color. reset invalidates history.
   Missing motion disables history for that call. Motion scales convert the
   caller's vector units into UV displacement; no sign or scale is guessed.
   Reprojection uses lmxxf's float-only fast five-tap filter, not its alternate
   fixed-point texture/reciprocal-table implementation. */
#define DLSS5_HIP_FRAME_MAGIC 0x3250494853534C44ULL
#define DLSS5_HIP_FRAME_ENV "DLSS5_HIP_FRAME_BRIDGE"
typedef struct Dlss5Frame {
    dlss5_u32 struct_size, flags;
    const float *rgba;
    float *rgb;
    const float *motion;
    dlss5_u32 motion_width, motion_height;
    float motion_uv_scale_x, motion_uv_scale_y;
    dlss5_u32 seed, reset;
    float paper_white, transfer, color;
} Dlss5Frame;
typedef struct Dlss5HipFrameBridge {
    unsigned long long magic;
    int (*run_frame)(const Dlss5Frame *frame);
} Dlss5HipFrameBridge;
DLSS5_CAPI_EXPORT int dlss5_run_frame(const Dlss5Frame *frame);

#define DLSS5_HIP_DEVICE_MAGIC 0x3349504853534C44ULL
#define DLSS5_HIP_DEVICE_ENV "DLSS5_HIP_DEVICE_BRIDGE"
typedef struct Dlss5HipDeviceBridge {
    unsigned long long magic;
    int (*find_device)(const char *exact_name);
} Dlss5HipDeviceBridge;
/* Unique exact name match only. Ambiguous or absent adapters return -1. */
DLSS5_CAPI_EXPORT int dlss5_find_device(const char *exact_name);

#ifdef __cplusplus
}
#endif
#endif
