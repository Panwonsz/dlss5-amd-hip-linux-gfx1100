#include <hip/hip_runtime.h>
#include <cstdio>

static bool fail_next;
static hipError_t test_malloc(void** pointer, size_t bytes) {
    if (fail_next) {
        fail_next = false;
        return hipErrorOutOfMemory;
    }
    return hipMalloc(pointer, bytes);
}
// Exercise the real DeviceBuf with one deterministic allocator failure.
#define hipMalloc test_malloc
#include "dlss5_runtime.hpp"
#undef hipMalloc

int main() {
    dlss5::DeviceBuf buffer(16);
    fail_next = true;
    bool rejected = false;
    try { buffer.alloc(32); }
    catch (const std::exception&) { rejected = true; }
    if (!rejected || buffer.p || buffer.bytes != 0) {
        std::fprintf(stderr, "failed allocation retained bytes=%zu pointer=%p\n", buffer.bytes, buffer.p);
        return 1;
    }
    buffer.alloc(32);
    float input[8] = {1,2,3,4,5,6,7,8}, output[8] = {};
    buffer.upload(input, sizeof input);
    buffer.download(output, sizeof output);
    for (unsigned i = 0; i < 8; ++i) if (output[i] != input[i]) return 1;
    buffer.reset();
    if (buffer.p || buffer.bytes) return 1;
    std::puts("DeviceBuf allocation failure/recovery: PASS");
    return 0;
}
