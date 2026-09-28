// All 65,536 E4M3 code pairs: the rescaling decode (DLSS5_E4M3_FASTDEC) against the integer recipe,
// bit for bit, on the GPU (the f16 multiply must keep denormals -- the HIP default).
//   hipcc -O3 -std=c++17 --offload-arch=gfx1100 hip/tests/test_e4m3_fastdec.hip -o /tmp/test-e4m3-fastdec
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
typedef uint32_t u32;
__device__ uint16_t dec_int(u32 b) {
    u32 e = (b >> 3) & 15u, m = b & 7u, s = (b & 0x80u) << 8;
    u32 normal = ((e + 8u) << 10) | (m << 7);
    u32 sub = (((0x88887760u >> (m * 4u)) & 15u) << 10) | (((0xE480u >> (m * 2u)) & 3u) << 8);
    u32 mask = 0u - u32(e != 0u);
    u32 nan = 0u - u32((b & 0x7fu) == 0x7fu);
    return uint16_t(s | (normal & mask) | (sub & ~mask) | (nan & 0x7e00u));
}
__device__ u32 dec_pair(u32 X) {
    typedef _Float16 h2_t __attribute__((ext_vector_type(2)));
    const u32 t = X & 0x007f007fu;
    const u32 bits = (t << 7) | ((X & 0x00800080u) << 8);
    h2_t v = __builtin_bit_cast(h2_t, bits);
    const h2_t k256 = {(_Float16)256.0f, (_Float16)256.0f};
    v = v * k256;
    const u32 nan = (((t + 0x00010001u) & 0x00800080u) >> 7) * 0x7e00u;
    return __builtin_bit_cast(u32, v) | nan;
}
__global__ void k(u32* bad) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 65536u) return;
    const u32 a = i & 0xffu, b = i >> 8;
    const u32 r = dec_pair(a | (b << 16));
    if ((r & 0xffffu) != dec_int(a) || (r >> 16) != dec_int(b)) atomicAdd(bad, 1u);
}
int main() {
    u32* d; u32 h = 0;
    if (hipMalloc(&d, 4) != hipSuccess || hipMemset(d, 0, 4) != hipSuccess) return 2;
    k<<<256, 256>>>(d);
    if (hipDeviceSynchronize() != hipSuccess || hipMemcpy(&h, d, 4, hipMemcpyDeviceToHost) != hipSuccess) return 2;
    std::printf("e4m3 fast decode: 65536 pairs, mismatches=%u -> %s\n", h, h ? "DO NOT USE" : "bit-identical");
    return h ? 1 : 0;
}
