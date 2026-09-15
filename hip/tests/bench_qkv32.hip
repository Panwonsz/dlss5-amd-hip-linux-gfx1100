// Isolated k_qkv_norm_f32<32,1> microbenchmark at real C32 stage sizes
// (full-res 1920x1152 = 2211840 tokens, half-res 960x576 = 552960 tokens).
// Reports ms/call and an output hash so barrier/instruction variants can be
// A/B checked for bit-identity without the full graph suite.
// BUILD ASM-VARIANT: -DQKV32_ASM_BARRIER swaps __syncthreads for a bare
// s_barrier signal/wait pair (no compiler-emitted global_inv scope:SCOPE_SE).
#include "dlss5_runtime.hpp"
#ifdef QKV32_ASM_BARRIER
__device__ inline void qkv32_barrier_asm() {
    asm volatile("s_barrier_signal -1\n\ts_barrier_wait -1\n\t" ::: "memory");
}
#define __syncthreads() qkv32_barrier_asm()
#endif
#include "kernels/qkv_norm.hip"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace dlss5;
static void qkv32_launch(const float* in, const u8* w, u8* out, uint m,
                         const float* scales, hipStream_t stream) {
    k_qkv_norm_f32<32, 1><<<dim3((m + 15) / 16, 3u), 64, 0, stream>>>(
        in, w, out, m, 32u, scales, 1.f, true);
}
int main(int argc, char** argv) {
    const uint c = 32, m = argc > 1 ? std::stoul(argv[1]) : 2211840u;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 12;
    std::vector<float> x(size_t(m) * c);
    for (size_t i = 0; i < x.size(); ++i) x[i] = H(float(int((i * 2654435761u) % 1000) - 500) * 0.002f);
    std::vector<float> w(3 * c * c);
    for (size_t i = 0; i < w.size(); ++i) w[i] = float(int((i * 40503u + i / 37u) % 97) - 48) * 0.01f;
    std::vector<u8> tiled(3 * c * c);
    pack_tiled_e4m3(tiled.data(), w.data(), 3 * c, c);
    std::vector<float> sc(c / 32);
    for (uint i = 0; i < sc.size(); ++i) sc[i] = H(float(int(i % 7) - 3) * 0.3f);
    DeviceBuf dx(x.size() * 4), dw(tiled.size()), ds(sc.size() * 4), dy(size_t(m) * 3 * c);
    dx.upload(x.data(), x.size() * 4); dw.upload(tiled.data(), tiled.size());
    ds.upload(sc.data(), sc.size() * 4);
    hipStream_t stream; hip_check(hipStreamCreate(&stream), "stream");
    auto go = [&] {
        qkv32_launch(dx.as<float>(), dw.as<u8>(), dy.as<u8>(), m, ds.as<float>(), stream);
    };
    for (int i = 0; i < 3; ++i) go();
    hip_check(hipStreamSynchronize(stream), "warm");
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0, stream);
    for (int i = 0; i < iters; ++i) go();
    hipEventRecord(e1, stream);
    hip_check(hipEventSynchronize(e1), "sync");
    float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= iters;
    std::printf("QKV32 m=%u iters=%d%s: %.3f ms/call\n", m, iters,
#ifdef QKV32_ASM_BARRIER
                " [asm-barrier]"
#else
                ""
#endif
                , ms);
    std::vector<u8> out(size_t(m) * 3 * c);
    dy.download(out.data(), out.size());
    uint64_t hsh = 1469598103934665603ull;
    for (u8 v : out) { hsh ^= v; hsh *= 1099511628211ull; }
    std::printf("out_hash=%016llx\n", (unsigned long long)hsh);
    return 0;
}
