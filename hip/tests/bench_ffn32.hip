// Isolated k_ffn_f32<32> microbenchmark: real C32 stage size (960x576 tokens),
// precise_c32 activation path. Reports ms/call and an output hash so source
// variants can be A/B checked for bit-identity without the full graph suite.
#include "dlss5_launch.hpp"
#include "dlss5_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace dlss5;
int main(int argc, char** argv) {
    const uint c = 32, tokens = argc > 1 ? std::stoul(argv[1]) : 552960u;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 20;
    std::vector<float> x(size_t(tokens) * c);
    for (size_t i = 0; i < x.size(); ++i) x[i] = H(float(int((i * 2654435761u) % 1000) - 500) * 0.002f);
    std::vector<float> w(8 * c * c);
    for (size_t i = 0; i < w.size(); ++i) w[i] = float(int((i * 40503u + i / 37u) % 97) - 48) * 0.01f;
    std::vector<float> sc(c);
    for (uint i = 0; i < c; ++i) sc[i] = H(float(int(i % 13) - 6) * 0.077f);
    std::vector<u8> tiled(w.size());
    pack_tiled_e4m3(tiled.data(), w.data(), 4 * c, c);
    pack_tiled_e4m3(tiled.data() + 4 * c * c, w.data() + 4 * c * c, c, 4 * c);
    DeviceBuf dx(x.size() * 4), dw(tiled.size()), ds(sc.size() * 4), dy(size_t(tokens) * c * 4);
    dx.upload(x.data(), x.size() * 4); dw.upload(tiled.data(), tiled.size());
    ds.upload(sc.data(), sc.size() * 4);
    hipStream_t stream; hip_check(hipStreamCreate(&stream), "stream");
    set_launch_stream(stream);
    auto go = [&] {
        launch_ffn_f32(c, dx.as<float>(), dw.as<u8>(), ds.as<float>(), dy.as<float>(), tokens, false, true);
    };
    for (int i = 0; i < 5; ++i) go();
    hip_check(hipStreamSynchronize(stream), "warm");
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0, stream);
    for (int i = 0; i < iters; ++i) go();
    hipEventRecord(e1, stream);
    hip_check(hipEventSynchronize(e1), "sync");
    float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= iters;
    std::printf("FFN32 tokens=%u iters=%d: %.3f ms/call\n", tokens, iters, ms);
    std::vector<float> out(size_t(tokens) * c);
    dy.download(out.data(), out.size() * 4);
    uint64_t hsh = 1469598103934665603ull;
    for (float v : out) { hsh ^= uint64_t(as_u32(v)); hsh *= 1099511628211ull; }
    std::printf("out_hash=%016llx min=%g max=%g\n", (unsigned long long)hsh,
                *std::min_element(out.begin(), out.end()), *std::max_element(out.begin(), out.end()));
    return 0;
}
