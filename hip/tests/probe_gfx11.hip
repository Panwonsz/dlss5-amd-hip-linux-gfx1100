// gfx11 diagnostics for the RDNA 3 port:
//  1) accumulator coordinate map: Get(i)/GetCoordinate(i) per lane vs. StoreF32 truth
//  2) WMMA magnitude: large (448*448) and tiny (1/512 * 1/512) products
//  3) f16-operand rounding: products needing >10 mantissa bits
// Build: hipcc -O3 -std=c++17 --offload-arch=gfx1100 -Iinclude -I/opt/rocm/include \
//        tests/probe_gfx11.hip -L/opt/rocm/lib -lamdhip64 -o probe_gfx11
#include "dlss5_linalg.hpp"
#include "dlss5_runtime.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace dlss5;
using namespace dlss5::linalg;

__global__ void k_probe_coord(const u8* A, const u8* B, float* X, float* got, uint* rc, uint* len) {
    A8 a = A8::Load(A, 0, 32);
    B8 b = B8::LoadCol(B, 0, 32);
    C32 c = Multiply(a, b);
    c.StoreF32(X, 0, 16 * 4);
    uint lane = threadIdx.x;
    len[lane] = c.Length();
    for (uint i = 0; i < c.Length() && i < 64u; i++) {
        uint2 p = c.GetCoordinate(i);
        got[lane * 64u + i] = c.Get(i);
        rc[lane * 64u + i] = p.x * 16u + p.y;
    }
}

__global__ void k_probe_mul(const u8* A, const u8* B, float* X) {
    A8 a = A8::Load(A, 0, 32);
    B8 b = B8::LoadCol(B, 0, 32);
    C32 c = Multiply(a, b);
    c.StoreF32(X, 0, 16 * 4);
}

static std::vector<float> run_mul(const std::vector<u8>& Ah, const std::vector<u8>& Bh) {
    DeviceBuf A(Ah.size()), B(Bh.size()), X(256 * 4);
    A.upload(Ah.data(), Ah.size());
    B.upload(Bh.data(), Bh.size());
    k_probe_mul<<<1, 32>>>(A.as<u8>(), B.as<u8>(), X.as<float>());
    hip_check(hipDeviceSynchronize(), "probe_mul");
    std::vector<float> x(256);
    X.download(x.data(), 256 * 4);
    return x;
}

static void cpu_gemm(const std::vector<u8>& A, const std::vector<u8>& B, std::vector<double>& C) {
    C.assign(256, 0.0);
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++) {
            double s = 0;
            float sf = 0;
            for (int k = 0; k < 32; k++) {
                s += double(from_e4m3(A[i * 32 + k])) * double(from_e4m3(B[j * 32 + k]));
                sf += from_e4m3(A[i * 32 + k]) * from_e4m3(B[j * 32 + k]);
            }
            C[i * 16 + j] = sf; // serial f32 order, like HLSL/gfx12
            (void)s;
        }
}

int main() {
    hip_check(hipSetDevice(0), "setDevice");
    std::printf("== probe_gfx11 ==\n");

    // ---- 1) coordinate map ----
    std::vector<u8> A(16 * 32, 0), B(16 * 32, 0);
    for (int r = 0; r < 16; r++) {
        A[r * 32 + 0] = e4m3_byte(float(r + 1));
        A[r * 32 + 1] = e4m3_byte(1.f);
    }
    for (int c = 0; c < 16; c++) {
        B[c * 32 + 0] = e4m3_byte(16.f);
        B[c * 32 + 1] = e4m3_byte(float(c + 1));
    }
    DeviceBuf dA(A.size()), dB(B.size()), dX(256 * 4), dG(32 * 64 * 4), dR(32 * 64 * 4), dL(32 * 4);
    dA.upload(A.data(), A.size());
    dB.upload(B.data(), B.size());
    k_probe_coord<<<1, 32>>>(dA.as<u8>(), dB.as<u8>(), dX.as<float>(), dG.as<float>(), dR.as<uint>(),
                             dL.as<uint>());
    hip_check(hipDeviceSynchronize(), "probe_coord");
    std::vector<float> X(256), G(32 * 64);
    std::vector<uint> R(32 * 64), L(32);
    dX.download(X.data(), 256 * 4);
    dG.download(G.data(), G.size() * 4);
    dR.download(R.data(), R.size() * 4);
    dL.download(L.data(), L.size() * 4);

    int store_bad = 0;
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++)
            if (X[r * 16 + c] != float(16 * (r + 1) + (c + 1)))
                store_bad++;
    std::printf("[1] StoreF32 truth mismatches=%d (expect 0)\n", store_bad);
    std::printf("[1] Length() per lane: lane0=%u lane16=%u lane31=%u\n", L[0], L[16], L[31]);
    int coord_bad = 0, shown = 0;
    std::printf("[1] lane i : actual(r,c)  acc_coord(r,c)\n");
    for (uint lane = 0; lane < 32; lane++)
        for (uint i = 0; i < L[lane] && i < 64u; i++) {
            float v = G[lane * 64 + i];
            int k = int(v) - 17;
            int ar = (k >= 0 && k < 256) ? k / 16 : -1, ac = (k >= 0 && k < 256) ? k % 16 : -1;
            int er = int(R[lane * 64 + i] / 16), ec = int(R[lane * 64 + i] % 16);
            bool bad = ar != er || ac != ec || float(k + 17) != v;
            coord_bad += bad;
            if (shown < 48 && (bad || lane < 2 || lane == 16)) {
                std::printf("    %2u %2u : (%2d,%2d) v=%g   (%2d,%2d)%s\n", lane, i, ar, ac, v, er, ec,
                            bad ? "  <-- MISMATCH" : "");
                shown++;
            }
        }
    std::printf("[1] coord mismatches=%d\n", coord_bad);

    // ---- 2) magnitude ----
    std::vector<u8> big(16 * 32, e4m3_byte(448.f)), tiny(16 * 32, 1u /* 1/512 */);
    auto xb = run_mul(big, big);
    auto xt = run_mul(tiny, tiny);
    std::printf("[2] big  448*448*32 = %.1f  gpu=%.1f\n", 448.0 * 448.0 * 32.0, xb[0]);
    std::printf("[2] tiny (1/512)^2*32 = %.10g  gpu=%.10g\n", 32.0 / (512.0 * 512.0), xt[0]);

    // ---- 3) random full-range E4M3, exact-equality vs serial f32 ----
    uint32_t s = 12345;
    auto rnd = [&]() { s = s * 1664525u + 1013904223u; return s >> 8; };
    int ranges[][2] = {{0, 0x77}, {0x00, 0x3f}, {0x38, 0x77}};
    for (auto& rg : ranges) {
        std::vector<u8> a(16 * 32), b(16 * 32);
        for (auto& x : a) x = u8(rg[0] + rnd() % (rg[1] - rg[0] + 1)) | u8((rnd() & 1) << 7);
        for (auto& x : b) x = u8(rg[0] + rnd() % (rg[1] - rg[0] + 1)) | u8((rnd() & 1) << 7);
        std::vector<double> ref;
        cpu_gemm(a, b, ref);
        auto g = run_mul(a, b);
        int neq = 0;
        double maxrel = 0;
        for (int i = 0; i < 256; i++) {
            if (g[i] != float(ref[i])) neq++;
            double d = std::fabs(g[i] - ref[i]) / (1e-30 + std::fabs(ref[i]));
            if (d > maxrel) maxrel = d;
        }
        std::printf("[3] codes 0x%02x..0x%02x: not-bit-equal=%d/256 max_rel_err=%.3g  sample gpu=%.9g cpu=%.9g\n",
                    rg[0], rg[1], neq, maxrel, g[5], ref[5]);
    }
    return 0;
}
