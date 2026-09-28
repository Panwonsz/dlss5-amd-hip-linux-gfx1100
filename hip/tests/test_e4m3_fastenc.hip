// All 2^32 f32 bit patterns: the fast E4M3 encoder (DLSS5_E4M3_FASTENC) against e4m3_byte_nb's
// integer recipe, bit for bit, on the GPU.
//   hipcc -O3 -std=c++17 --offload-arch=gfx1100 hip/tests/test_e4m3_fastenc.hip -o /tmp/test-e4m3-fastenc
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
typedef uint32_t u32;
__device__ inline u32 mk(bool c) { return 0u - u32(c); }
__device__ unsigned char enc_int(float v) {
    u32 b = __float_as_uint(v), a = b & 0x7fffffffu, sg = (b >> 24) & 0x80u;
    u32 q = u32(nearbyintf(fminf(fabsf(v), 1.f) * 512.f));
    u32 r = (a + 0x7ffffu + ((a >> 20) & 1u)) & 0xfff00000u;
    u32 cn = ((r >> 23) - 120u) * 8u + ((r >> 20) & 7u);
    u32 mc = mk(cn > 126u);
    cn = (cn & ~mc) | (126u & mc);
    u32 ms = mk(a < 0x3c800000u), mt = mk(a >= 0x43e00000u), mn = mk(a > 0x7f800000u);
    u32 code = (q & ms) | (cn & ~ms);
    code = (code & ~mt) | (126u & mt);
    code = (code & ~mn) | (127u & mn);
    return (unsigned char)(sg | code);
}
__device__ unsigned char enc_fast(float v) {
    const u32 b = __float_as_uint(v), a = b & 0x7fffffffu, sg = (b >> 24) & 0x80u;
    const u32 r = (a + 0x7ffffu + ((a >> 20) & 1u)) & 0xfff00000u;
    const u32 cn = min((r >> 20) - 960u, 126u);
    const u32 q = __float_as_uint(fabsf(v) + 16384.f) - 0x46800000u;
    u32 code = a < 0x3c800000u ? q : cn;
    code = a > 0x7f800000u ? 127u : code;
    return (unsigned char)(sg | code);
}
__global__ void k(u32 hi, unsigned long long* bad, u32* first) {
    const u32 bits = (hi << 24) | (blockIdx.x * blockDim.x + threadIdx.x);
    const float v = __uint_as_float(bits);
    if (enc_int(v) != enc_fast(v)) { atomicAdd(bad, 1ull); atomicMin(first, bits); }
}
int main() {
    unsigned long long* d; u32* f; unsigned long long h = 0; u32 fh = 0;
    if (hipMalloc(&d, 8) != hipSuccess || hipMalloc(&f, 4) != hipSuccess) return 2;
    if (hipMemset(d, 0, 8) != hipSuccess || hipMemset(f, 0xff, 4) != hipSuccess) return 2;
    for (u32 hi = 0; hi < 256u; ++hi) k<<<(1u << 24) / 256u, 256>>>(hi, d, f);
    if (hipDeviceSynchronize() != hipSuccess) return 2;
    if (hipMemcpy(&h, d, 8, hipMemcpyDeviceToHost) != hipSuccess || hipMemcpy(&fh, f, 4, hipMemcpyDeviceToHost) != hipSuccess) return 2;
    std::printf("e4m3 fast encode: 4294967296 inputs, mismatches=%llu", h);
    if (h) std::printf(" (first 0x%08x)", fh);
    std::printf(" -> %s\n", h ? "DO NOT USE" : "bit-identical");
    return h ? 1 : 0;
}
