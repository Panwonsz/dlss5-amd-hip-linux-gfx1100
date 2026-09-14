// Experiment: hardware v_cvt_pk_fp8_f32 (OCP E4M3 RNE) vs software e4m3_byte.
// Compares three variants per value: software (current), raw HW cvt, and
// HW cvt with finite pre-clamp to +/-448 (HIP header strategy). Reports the
// first mismatches so we can pick a bit-exact hardware variant or abort.
#include "dlss5_common.hpp"
#include "dlss5_runtime.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace dlss5;

__global__ void probe(const float* x, u32* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    u8 sw = e4m3_byte(v);
    uint16_t p = __builtin_amdgcn_cvt_pk_fp8_f32(v, 0.f, 0, false);
    u8 raw = (u8)(p & 0xffu);
    u8 hi = (u8)(p >> 8);
    u32 b = as_u32(v);
    float vc = ((b & 0x7f800000u) != 0x7f800000u) ? __builtin_amdgcn_fmed3f(v, 448.f, -448.f) : v;
    uint16_t p2 = __builtin_amdgcn_cvt_pk_fp8_f32(vc, 0.f, 0, false);
    u8 clamped = (u8)(p2 & 0xffu);
    // final variant: early SW return for non-finite, branchless fmed3f clamp for finite
    u8 fin = (b >= 0x7f800000u) ? sw
        : (u8)(__builtin_amdgcn_cvt_pk_fp8_f32(__builtin_amdgcn_fmed3f(v, 448.f, -448.f), 0.f, 0, false) & 0xffu);
    out[i] = u32(sw) | u32(raw) << 8 | u32(clamped) << 16 | u32(fin) << 24;
}

__global__ void order_probe(u32* out) {
    if (blockIdx.x * blockDim.x + threadIdx.x) return;
    out[0] = __builtin_amdgcn_cvt_pk_fp8_f32(1.f, -1.f, 0, false);
}

int main() {
    std::vector<float> x;
    auto push = [&](float v) { x.push_back(v); };
    // 1. all 256 representable e4m3 values
    for (int code = 0; code < 256; ++code) push(from_e4m3(u8(code)));
    // 2. every midpoint (RNE tie) between adjacent representable values
    for (int code = 0; code < 255; ++code) {
        double a = double(from_e4m3(u8(code))), b = double(from_e4m3(u8(code + 1)));
        if (a == b) continue;
        push(float((a + b) / 2.0));
    }
    // 3. dense scan around 0 and the denormal range
    for (int i = -8192; i <= 8192; ++i) push(float(i) * 0x1p-16f);
    for (int i = -64; i <= 64; ++i) push(float(i) * 0x1p-14f);
    // 4. edge constants
    push(0.f); push(-0.f);
    push(448.f); push(-448.f); push(447.99f); push(448.001f);
    push(449.f); push(500.f); push(1e3f); push(1e30f);
    push(INFINITY); push(-INFINITY);
    push(as_f32(0x7f800001u)); push(as_f32(0x7fffffffu));
    push(as_f32(0xff800001u)); push(as_f32(0xffffffffu));
    push(as_f32(0x00000001u)); push(as_f32(0x80000001u)); // f32 subnormals
    push(1e-38f); push(-1e-38f);
    // 5. random: log-uniform magnitude 1e-38..1e30 + uniform [-1,1]
    srand(12345);
    for (int i = 0; i < 2000000; ++i) {
        double e = 1e-38 * std::pow(1e30 / 1e-38, double(rand()) / RAND_MAX);
        float v = float(e) * (rand() & 1 ? 1.f : -1.f);
        push(v);
    }
    for (int i = 0; i < 2000000; ++i)
        push((float(rand()) / RAND_MAX - 0.5f) * 2.f);
    const int n = (int)x.size();

    std::vector<u32> order(1);
    DeviceBuf dx(size_t(n) * 4), dy(n * 4), do_(4);
    dx.upload(x.data(), size_t(n) * 4);
    do_.upload(order.data(), 4);
    order_probe<<<1, 1>>>((u32*)do_.as<u32>());
    probe<<<(n + 255) / 256, 256>>>((const float*)dx.as<float>(), (u32*)dy.as<u32>(), n);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("device error: %s\n", hipGetErrorString(e)); return 2; }
    do_.download(order.data(), 4);
    printf("order_probe(1.0,-1.0) = 0x%04x (expect 0xb838 if low=x)\n", order[0]);
    std::vector<u32> y(n);
    dy.download(y.data(), size_t(n) * 4);

    long sw_raw = 0, sw_cl = 0, sw_fin = 0;
    for (int i = 0; i < n; ++i) {
        u8 sw = y[i] & 0xff, raw = (y[i] >> 8) & 0xff, cl = (y[i] >> 16) & 0xff, fin = (y[i] >> 24) & 0xff;
        if (sw != raw && sw_raw < 3) printf("RAW  mismatch i=%d v=%g sw=0x%02x hw=0x%02x\n", i, x[i], sw, raw);
        if (sw != raw) ++sw_raw;
        if (sw != cl && sw_cl < 3) printf("CLMP mismatch i=%d v=%g sw=0x%02x hw=0x%02x\n", i, x[i], sw, cl);
        if (sw != cl) ++sw_cl;
        if (sw != fin && sw_fin < 10) printf("FINAL mismatch i=%d v=%g (0x%08x) sw=0x%02x fin=0x%02x\n", i, x[i], as_u32(x[i]), sw, fin);
        if (sw != fin) ++sw_fin;
    }
    printf("total=%d  sw_vs_raw=%ld  sw_vs_clamped=%ld  sw_vs_FINAL=%ld\n", n, sw_raw, sw_cl, sw_fin);
    return (sw_fin == 0) ? 0 : 1;
}
