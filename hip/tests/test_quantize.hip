#include "dlss5_common.hpp"
#include "dlss5_runtime.hpp"
#include <cstdio>
#include <limits>
#include <vector>

using namespace dlss5;

static float decode(unsigned b) {
    unsigned e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7)
        return std::numeric_limits<float>::quiet_NaN();
    float v = e ? std::ldexp(1.f + float(m) / 8.f, int(e) - 7) : float(m) / 512.f;
    return b & 128 ? -v : v;
}

static u8 nearest(float v) {
    unsigned sign = std::signbit(v) ? 128 : 0;
    if (std::isnan(v)) return u8(sign | 127);
    if (std::isinf(v)) return u8(sign | 126);
    float a = std::fabs(v), distance = std::numeric_limits<float>::infinity();
    unsigned best = 0;
    for (unsigned b = 0; b < 127; ++b) {
        float d = std::fabs(a - decode(b));
        if (d < distance || (d == distance && !(b & 1))) {
            distance = d;
            best = b;
        }
    }
    return u8(sign | best);
}

__global__ void convert(const float* in, u8* out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = e4m3_byte(in[i]);
}

int main() {
    std::vector<float> values{0.f, -0.f, 1.1875f, 1.9375f, 0.0146484375f,
                             1000.f, -1000.f, INFINITY, -INFINITY, NAN};
    for (unsigned b = 0; b < 126; ++b) {
        float mid = (decode(b) + decode(b + 1)) * .5f;
        for (float v : {decode(b), std::nextafter(mid, 0.f), mid,
                        std::nextafter(mid, INFINITY)}) {
            values.push_back(v);
            values.push_back(-v);
        }
    }
    unsigned bad = 0;
    for (float v : values) {
        if (e4m3_byte(v) != nearest(v)) ++bad;
    }
    unsigned decode_bad = 0;
    for (unsigned b = 0; b < 256; ++b) {
        float expected = decode(b), actual = from_e4m3(u8(b));
        if (std::isnan(expected) ? !std::isnan(actual)
                                : actual != expected || std::signbit(actual) != std::signbit(expected))
            ++decode_bad;
    }
    DeviceBuf in(values.size() * sizeof(float)), out(values.size());
    in.upload(values.data(), values.size() * sizeof(float));
    convert<<<(values.size()+255)/256,256>>>(in.as<float>(),out.as<u8>(),unsigned(values.size()));
    hip_check(hipDeviceSynchronize(), "quantize");
    std::vector<u8> gpu(values.size());
    out.download(gpu.data(), gpu.size());
    unsigned gpu_bad = 0;
    for (size_t i = 0; i < values.size(); ++i)
        if (gpu[i] != nearest(values[i])) ++gpu_bad;
    std::printf("E4M3 RNE samples=%zu host_bad=%u gpu_bad=%u decode_bad=%u\n",
                values.size(),bad,gpu_bad,decode_bad);
    return bad || gpu_bad || decode_bad ? 1 : 0;
}
