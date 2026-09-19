// gfx11 microbenchmark: is the per-load E4M3->f16 decode the bottleneck?
// Same 16x32 A . 32x16 B multiply, three ways, over many blocks:
//   f8    : A8/B8 (E4M3 bytes, decoded to f16 at every load - current gfx11 path)
//   f16   : A16/B16 (native f16 buffers, no decode)
//   store : f16 multiply + element-wise E4M3 store (StoreF8), like most kernels
// Build: hipcc -O3 -std=c++17 --offload-arch=gfx1100 -Iinclude -I/opt/rocm/include \
//        tests/bench_gfx11_load.hip -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib -o bench_gfx11_load
#include "dlss5_linalg.hpp"
#include "dlss5_runtime.hpp"
#include <chrono>
#include <cstdio>
#include <vector>

using namespace dlss5;
using namespace dlss5::linalg;

__global__ void k_f8(const u8* A, const u8* B, float* X) {
    uint o = blockIdx.x;
    A8 a = A8::Load(A, o * 512u, 32);
    B8 b = B8::LoadCol(B, o * 512u, 32);
    C32 c = Multiply(a, b);
    c.StoreF32(X, o * 1024u, 16 * 4);
}
__global__ void k_f16(const f16* A, const f16* B, float* X) {
    uint o = blockIdx.x;
    A16 a = A16::Load(A, o * 1024u, 64);
    B16 b = B16::LoadCol(B, o * 1024u, 64);
    C32 c = Multiply(a, b);
    c.StoreF32(X, o * 1024u, 16 * 4);
}
__global__ void k_store8(const f16* A, const f16* B, u8* Y) {
    uint o = blockIdx.x;
    A16 a = A16::Load(A, o * 1024u, 64);
    B16 b = B16::LoadCol(B, o * 1024u, 64);
    C32 c = Multiply(a, b);
    c.StoreF8(Y, o * 256u, 16);
}

int main() {
    hip_check(hipSetDevice(0), "setDevice");
    const uint blocks = 1u << 14; // 16384 multiplies = 67M scalar products
    std::vector<u8> a8(blocks * 512ull), b8(blocks * 512ull);
    std::vector<f16> a16(blocks * 512ull), b16(blocks * 512ull);
    uint32_t s = 7;
    for (size_t i = 0; i < a8.size(); i++) {
        s = s * 1664525u + 1013904223u;
        a8[i] = u8((s >> 8) % 0x60);
        b8[i] = u8((s >> 16) % 0x60);
        a16[i] = f16(from_e4m3(a8[i]));
        b16[i] = f16(from_e4m3(b8[i]));
    }
    DeviceBuf dA8(a8.size()), dB8(b8.size()), dA16(a16.size() * 2), dB16(b16.size() * 2),
              dX(blocks * 1024ull), dY(blocks * 256ull);
    dA8.upload(a8.data(), a8.size()); dB8.upload(b8.data(), b8.size());
    dA16.upload(a16.data(), a16.size() * 2); dB16.upload(b16.data(), b16.size() * 2);

    using clk = std::chrono::steady_clock;
    auto time = [&](const char* name, auto&& fn) {
        fn(); hip_check(hipDeviceSynchronize(), name); // warm-up
        auto t0 = clk::now();
        const int reps = 5;
        for (int r = 0; r < reps; r++) fn();
        hip_check(hipDeviceSynchronize(), name);
        double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / reps;
        double gflops = double(blocks) * 16 * 32 * 16 / (ms / 1000.0) / 1e9;
        std::printf("%-6s %8.1f ms  %7.2f GFLOPS\n", name, ms, gflops);
    };
    time("f8", [&] { k_f8<<<blocks, 32>>>(dA8.as<u8>(), dB8.as<u8>(), dX.as<float>()); });
    time("f16", [&] { k_f16<<<blocks, 32>>>(dA16.as<f16>(), dB16.as<f16>(), dX.as<float>()); });
    time("store", [&] { k_store8<<<blocks, 32>>>(dA16.as<f16>(), dB16.as<f16>(), dY.as<u8>()); });
    return 0;
}
