// Classifying a GPU/reference mismatch: is it one representable step, or a real disagreement?
//
// These tests compare against a reference that accumulates in a different order from the kernel --
// double-precision with the residual first in test_linear, serial-over-k in test_ffn_f32, where the
// kernel sums WMMA tiles and adds the residual last. On gfx1201 those agree exactly, because FP8
// WMMA accumulates onto the residual in a single rounding. On gfx11 there is no FP8 matrix
// instruction: f16 products accumulate into f32 in a different order, the two answers differ by
// well under an f32 ulp, and the final quantization to f16 or E4M3 turns that into one whole step
// whenever it lands on a rounding tie.
//
// A flat absolute threshold cannot tell those apart from a genuine bug, which is why both tests
// reported dozens of failures on every gfx1100 build the repository has ever had. So measure the
// distance in the OUTPUT'S OWN grid instead, against a per-grid budget (see ulp_budget below).
//
// This is not a loosened tolerance:
//   - anything past the budget fails on every architecture, as does any non-finite value;
//   - divergences within it are counted and printed, so a regression shows up as the number moving;
//   - `worst_steps` is printed unconditionally, so drift is visible even on a passing run;
//   - off gfx11 the budget is not granted at all -- on gfx1201 the two orders do agree exactly, so
//     a single step there is a failure.
#pragma once
#include "dlss5_common.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>

namespace dlss5 {

enum class Grid { F32, F16, E4M3 };

// Sign-magnitude bits mapped to a monotonic signed key, so adjacent representable values differ by
// one. The usual total-ordering trick; ties at +-0 collapse to the same key, which is what we want.
inline long long ulp_key(float v, Grid g) {
    switch (g) {
    case Grid::F16: {
        // __half_as_ushort is device-only; this header runs on the host, so read the bits out of
        // the __half directly. (It is still parsed by the device pass, which is why it cannot just
        // call the device intrinsic.)
        __half h = __float2half(v);
        uint16_t b;
        std::memcpy(&b, &h, sizeof(b));
        return (b & 0x8000u) ? -(long long)(b & 0x7fffu) : (long long)(b & 0x7fffu);
    }
    case Grid::E4M3: {
        u8 b = e4m3_byte(v);
        return (b & 0x80u) ? -(long long)(b & 0x7fu) : (long long)(b & 0x7fu);
    }
    default: {
        u32 b = as_u32(v);
        return (b & 0x80000000u) ? -(long long)(b & 0x7fffffffu) : (long long)(b & 0x7fffffffu);
    }
    }
}

inline long long ulp_steps(float got, float want, Grid g) {
    long long d = ulp_key(got, g) - ulp_key(want, g);
    return d < 0 ? -d : d;
}

inline bool device_is_gfx11() {
    int d = 0;
    hipDeviceProp_t p{};
    if (hipGetDevice(&d) != hipSuccess) return false;
    if (hipGetDeviceProperties(&p, d) != hipSuccess) return false;
    return std::string(p.gcnArchName).rfind("gfx11", 0) == 0;
}

// How many steps the accumulation order can move a result, per grid.
//
// F16 and E4M3 are the quantized outputs: the kernel and the reference differ by far less than an
// f32 ulp before the final rounding, so the only way that becomes visible is a value sitting on a
// rounding tie and falling the other way. That is exactly one step, never two.
//
// F32 is the unrounded accumulator (mode 3), and a one-step rule would be wrong there: the kernel
// sums 96 products in WMMA tile order against a reference that sums them in index order, and
// reordered floating-point summation is allowed to drift by O(n)*eps. A handful of ulps is expected
// and is not evidence of anything. `worst_steps` is printed on every run, so the number to watch is
// that, not this budget -- if it starts climbing, something has changed even while the test passes.
inline long long ulp_budget(Grid g) { return g == Grid::F32 ? 4 : 1; }

struct UlpTally {
    unsigned hard = 0;   // non-finite, or past the budget: a defect anywhere
    unsigned soft = 0;   // within budget: the documented gfx11 accumulation-order difference
    long long worst = 0;
    float max_abs = 0.f;

    void check(float got, float want, Grid g) {
        float e = std::fabs(got - want);
        if (e > max_abs) max_abs = e;
        if (!std::isfinite(got)) { ++hard; return; }
        long long s = ulp_steps(got, want, g);
        if (s > worst) worst = s;
        if (s == 0) return;
        if (s <= ulp_budget(g)) ++soft; else ++hard;
    }

    // Exit status. One step is tolerated only where the accumulation orders genuinely differ.
    int report(const char* what) const {
        const bool gfx11 = device_is_gfx11();
        std::printf("%s: mismatches=%u one_step=%u worst_steps=%lld max_abs=%g%s\n",
                    what, hard, soft, worst, max_abs,
                    soft && gfx11 ? "  [gfx11 accumulation order, see ulp_compare.hpp]" : "");
        if (hard) return 1;
        if (soft && !gfx11) {
            std::printf("%s: one-step divergence is not expected off gfx11 -- FAIL\n", what);
            return 1;
        }
        return 0;
    }
};

} // namespace dlss5
