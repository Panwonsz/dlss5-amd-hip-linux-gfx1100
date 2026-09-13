/* Private local lmxxf/vkd3d extension. No graphics waits in API callers.
 * The recording caller owns the initial context ref. Accepted markers add a ref
 * held by the command allocator until its GPU work completes/reset is legal.
 * run executes on vkd3d's submission worker AFTER the prefix timeline completes,
 * with the Vulkan queue mutex released, BEFORE the suffix is submitted.
 * run must NOT submit D3D work or wait on any suffix. A nonzero result loses the
 * device rather than consuming unwritten memory. Callbacks use Win64 ABI.
 * Unsupported runtimes leave accepted=0, letting the caller omit writeback.
 */
#ifndef DLSS5_SUBMIT_H
#define DLSS5_SUBMIT_H
#include <stdint.h>
#define DLSS5_SUBMIT_METADATA 0x3548504cu
#define DLSS5_SUBMIT_MAGIC UINT64_C(0x3154494d4255534c)
#define DLSS5_SUBMIT_VERSION 1u
#if defined(_WIN32)
#define DLSS5_PE_ABI
#else
#define DLSS5_PE_ABI __attribute__((ms_abi))
#endif
#ifdef __cplusplus
extern "C" {
#endif
struct Dlss5SubmitMarker {
    uint64_t magic;
    uint32_t version, bytes;
    void *context;
    void (DLSS5_PE_ABI *retain)(void *);
    void (DLSS5_PE_ABI *release)(void *);
    int (DLSS5_PE_ABI *run)(void *, uint64_t queue_identity);
    uint32_t *accepted; /* recording-time only; never saved or touched by worker */
};
#ifdef __cplusplus
}
static_assert(sizeof(Dlss5SubmitMarker)==56,"x64 submission marker ABI");
#endif
#endif
