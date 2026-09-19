// gfx11: map rocWMMA fragment element indices to matrix (row, col), then benchmark a
// load that fills fragments element-wise straight from E4M3 bytes (no private f16 array).
// Build: hipcc -O3 -std=c++17 --offload-arch=gfx1100 -Iinclude -I/opt/rocm/include \
//        tests/probe_frag.hip -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib -o probe_frag
#include "dlss5_linalg.hpp"
#include "dlss5_runtime.hpp"
#include <chrono>
#include <cstdio>
#include <vector>

using namespace dlss5;
using namespace dlss5::linalg;

using FA  = FragA<f16, row_major>;
using FBR = FragB<f16, row_major>;
using FBC = FragB<f16, col_major>;
constexpr uint MAXE = 64;

// kind 0: A row_major, 1: B row_major, 2: B col_major. Buffer holds v = index (0..255).
template <typename Frag>
__device__ void record(const f16* buf, uint* out, uint* len, uint kind) {
    Frag f;
    rocwmma::load_matrix_sync(f, buf, 16u);
    uint lane = threadIdx.x;
    len[kind * 32u + lane] = Frag::num_elements;
    for (uint i = 0; i < Frag::num_elements && i < MAXE; i++)
        out[(kind * 32u + lane) * MAXE + i] = uint(float(f[i]));
}
__global__ void k_record(const f16* buf, uint* out, uint* len) {
    record<FA>(buf, out, len, 0);
    record<FBR>(buf, out, len, 1);
    record<FBC>(buf, out, len, 2);
}

// Table-driven element fill: tab[(kind*32+lane)*MAXE+i] = lead<<8 | trail, memory index = lead*ldm + trail
// (row_major: lead=row, col_major: lead=col).
template <typename Frag>
__device__ inline void fill_e4m3(Frag& f, const u8* p, uint ldm, const uint* tab, uint kind) {
    const uint base = (kind * 32u + (threadIdx.x & 31u)) * MAXE;
    for (uint i = 0; i < Frag::num_elements; i++) {
        uint rc = tab[base + i];
        f[i] = f16(from_e4m3(p[(rc >> 8) * ldm + (rc & 255u)]));
    }
}
__global__ void k_f8map(const u8* A, const u8* B, float* X, const uint* tab) {
    uint o = blockIdx.x;
    const u8* a = A + o * 512u;
    const u8* b = B + o * 512u;
    FA a0, a1;
    FBC b0, b1;
    // A row_major 16x32 stride 32: k0 at col 0, k1 at col 16.
    fill_e4m3(a0, a, 32u, tab, 0);
    fill_e4m3(a1, a + 16, 32u, tab, 0);
    // B col_major 32x16 stride 32: element (r, c) = p[c*32 + r]; k1 rows start at +16.
    fill_e4m3(b0, b, 32u, tab, 2);
    fill_e4m3(b1, b + 16, 32u, tab, 2);
    FragC acc;
    rocwmma::fill_fragment(acc, 0.f);
    rocwmma::mma_sync(acc, a0, b0, acc);
    rocwmma::mma_sync(acc, a1, b1, acc);
    rocwmma::store_matrix_sync(X + o * 256u, acc, 16u, rocwmma::mem_row_major);
}
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

int main() {
    hip_check(hipSetDevice(0), "setDevice");
    // ---- 1) map ----
    std::vector<f16> idx(256);
    for (int i = 0; i < 256; i++) idx[i] = f16(float(i));
    DeviceBuf dI(512), dO(3 * 32 * MAXE * 4), dL(3 * 32 * 4);
    dI.upload(idx.data(), 512);
    k_record<<<1, 32>>>(dI.as<f16>(), dO.as<uint>(), dL.as<uint>());
    hip_check(hipDeviceSynchronize(), "record");
    std::vector<uint> O(3 * 32 * MAXE), L(3 * 32), tab(3 * 32 * MAXE, 0);
    dO.download(O.data(), O.size() * 4);
    dL.download(L.data(), L.size() * 4);
    const char* names[] = {"A row_major", "B row_major", "B col_major"};
    for (uint k = 0; k < 3; k++) {
        std::printf("[%s] num_elements/lane=%u\n", names[k], L[k * 32]);
        std::vector<int> seen(256, 0);
        for (uint lane = 0; lane < 32; lane++) {
            if (lane < 2 || lane == 16 || lane == 17) std::printf("  lane %2u:", lane);
            for (uint i = 0; i < L[k * 32 + lane] && i < MAXE; i++) {
                uint v = O[(k * 32 + lane) * MAXE + i];
                uint r = k == 2 ? v % 16 : v / 16, c = k == 2 ? v / 16 : v % 16;
                if (v < 256) seen[v]++;
                tab[(k * 32 + lane) * MAXE + i] = ((v / 16) << 8) | (v % 16); // lead*ldm + trail
                if (lane < 2 || lane == 16 || lane == 17)
                    if (i < 8) std::printf(" (%u,%u)", r, c);
            }
            if (lane < 2 || lane == 16 || lane == 17) std::printf("\n");
        }
        int missing = 0, dup = 0;
        for (int v : seen) { missing += v == 0; dup += v > 1; }
        std::printf("  coverage: missing=%d duplicated=%d\n", missing, dup);
    }

    // ---- 2) correctness + speed ----
    const uint blocks = 1u << 16;
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
              dX1(blocks * 256ull * 4), dX2(blocks * 256ull * 4), dT(tab.size() * 4);
    dA8.upload(a8.data(), a8.size()); dB8.upload(b8.data(), b8.size());
    dA16.upload(a16.data(), a16.size() * 2); dB16.upload(b16.data(), b16.size() * 2);
    dT.upload(tab.data(), tab.size() * 4);

    using clk = std::chrono::steady_clock;
    auto time = [&](const char* name, auto&& fn) {
        fn(); hip_check(hipDeviceSynchronize(), name);
        const int reps = 10;
        auto t0 = clk::now();
        for (int r = 0; r < reps; r++) fn();
        hip_check(hipDeviceSynchronize(), name);
        double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / reps;
        std::printf("%-8s %9.2f ms  %8.1f GFLOPS\n", name, ms, double(blocks) * 16 * 32 * 16 / (ms / 1e3) / 1e9);
    };
    time("f8", [&] { k_f8<<<blocks, 32>>>(dA8.as<u8>(), dB8.as<u8>(), dX1.as<float>()); });
    time("f16", [&] { k_f16<<<blocks, 32>>>(dA16.as<f16>(), dB16.as<f16>(), dX1.as<float>()); });
    time("f8map", [&] { k_f8map<<<blocks, 32>>>(dA8.as<u8>(), dB8.as<u8>(), dX2.as<float>(), dT.as<uint>()); });
    std::vector<float> x1(blocks * 256ull), x2(blocks * 256ull);
    k_f8<<<blocks, 32>>>(dA8.as<u8>(), dB8.as<u8>(), dX1.as<float>());
    hip_check(hipDeviceSynchronize(), "f8 ref");
    dX1.download(x1.data(), x1.size() * 4);
    dX2.download(x2.data(), x2.size() * 4);
    size_t bad = 0;
    for (size_t i = 0; i < x1.size(); i++) bad += x1[i] != x2[i];
    std::printf("f8map vs f8 result mismatches=%zu / %zu\n", bad, x1.size());
    return 0;
}
