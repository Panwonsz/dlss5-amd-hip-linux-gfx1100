#pragma once
#include "dlss5_runtime.hpp"
#include <memory>
#include <array>

namespace dlss5 {

struct Network70 {
    Network70();
    ~Network70();
    // Fixed native geometry: input 1920x1080 RGBA F32, output 1920x1152 RGB F32.
    // create is transactional and repeatable; do not race create/destruction with
    // methods on this same object. Runs on one object are serialized internally.
    void create(const std::string& dir);
    // No stream capture: every run consumes the supplied seed. All intermediate
    // activations remain device-resident. DLSS5_HIP_TRACE=1 adds 71 GPU reductions
    // and reports them after completion; it is intentionally expensive.
    // Optional already-warped/reflected 1920x1152 raster RGBA F32 history.
    // nullptr disables history for this call; no history is implicitly retained.
    void run(const float* host_rgb, float* host_out, uint seed, const float* host_history = nullptr);
    float last_gpu_ms() const;
    struct Trace { uint elements{}, nonfinite{}, hash{}; float minimum{}, maximum{}; double sum{}; bool checked{}; };
    std::array<uint,71> last_block_counts() const;
    std::array<Trace,71> last_trace() const;
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace dlss5
