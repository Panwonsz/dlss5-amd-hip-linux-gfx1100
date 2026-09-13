#include "dlss5_linalg.hpp"
#include "dlss5_runtime.hpp"
#include <vector>
#include <cstdio>
#include <cmath>
#include <random>

using namespace dlss5;
using namespace dlss5::linalg;

__global__ void k_mma_f8(const u8* A, const u8* B, float* C) {
    A8 a = A8::Load(A, 0, 32);
    B8 b = B8::LoadCol(B, 0, 32);
    C32 c = Multiply(a, b);
    c.StoreF32(C, 0, 16 * 4);
}

static void cpu_gemm(const u8* A, const u8* B, float* C) {
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++) {
            float s = 0;
            for (int k = 0; k < 32; k++)
                s += from_e4m3(A[i * 32 + k]) * from_e4m3(B[j * 32 + k]);
            C[i * 16 + j] = s;
        }
}

int main() {
    hip_check(hipSetDevice(0), "setDevice");
    hipDeviceProp_t prop{};
    hip_check(hipGetDeviceProperties(&prop, 0), "props");
    std::printf("device=%s arch=%d.%d\n", prop.name, prop.major, prop.minor);

    std::mt19937 rng(1);
    std::uniform_int_distribution<int> dist(0, 0x77); // e=0..14, no OCP NaN (e=15)
    std::vector<u8> Ah(16 * 32), Bh(16 * 32);
    for (auto& x : Ah)
        x = u8(dist(rng));
    for (auto& x : Bh)
        x = u8(dist(rng));
    std::vector<float> cpu(256), gpu(256);
    cpu_gemm(Ah.data(), Bh.data(), cpu.data());

    DeviceBuf A(Ah.size()), B(Bh.size()), C(256 * 4);
    A.upload(Ah.data(), Ah.size());
    B.upload(Bh.data(), Bh.size());
    k_mma_f8<<<1, 32>>>(A.as<u8>(), B.as<u8>(), C.as<float>());
    hip_check(hipGetLastError(), "k_mma_f8");
    hip_check(hipDeviceSynchronize(), "sync");
    C.download(gpu.data(), 256 * 4);

    int bad = 0;
    float maxe = 0;
    for (int i = 0; i < 256; i++) {
        float e = fabsf(gpu[i] - cpu[i]);
        maxe = fmaxf(maxe, e);
        if (!(e <= 1e-2f * (1.f + fabsf(cpu[i]))))
            bad++;
    }
    std::printf("wmma_f8 16x32x16 max_abs_err=%.6f mismatches=%d %s\n", maxe, bad,
                bad ? "FAIL" : "ok");
    if (bad) {
        for (int i = 0; i < 16 && i < 4; i++) {
            std::printf(" row %d gpu:", i);
            for (int j = 0; j < 4; j++)
                std::printf(" %.4f", gpu[i * 16 + j]);
            std::printf(" cpu:");
            for (int j = 0; j < 4; j++)
                std::printf(" %.4f", cpu[i * 16 + j]);
            std::printf("\n");
        }
        return 1;
    }
    return 0;
}
