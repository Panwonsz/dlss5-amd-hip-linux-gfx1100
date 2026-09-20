#pragma once
#include "dlss5_common.hpp"
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdio>

namespace dlss5 {

inline void hip_check(hipError_t e, const char* what) {
    if (e != hipSuccess)
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

struct DeviceBuf {
    void* p{};
    size_t bytes{};
    DeviceBuf() = default;
    explicit DeviceBuf(size_t n) { alloc(n); }
    DeviceBuf(const DeviceBuf&) = delete;
    DeviceBuf& operator=(const DeviceBuf&) = delete;
    DeviceBuf(DeviceBuf&& o) noexcept : p(o.p), bytes(o.bytes) { o.p = nullptr; o.bytes = 0; }
    DeviceBuf& operator=(DeviceBuf&& o) noexcept {
        if (this != &o) {
            reset();
            p = o.p;
            bytes = o.bytes;
            o.p = nullptr;
            o.bytes = 0;
        }
        return *this;
    }
    ~DeviceBuf() { reset(); }
    void alloc(size_t n) {
        reset();
        if (n) {
            void* next = nullptr;
            hip_check(hipMalloc(&next, n), "hipMalloc");
            p = next;
            bytes = n;
        }
    }
    void reset() {
        if (p)
            (void)hipFree(p);
        p = nullptr;
        bytes = 0;
    }
    template <class T>
    T* as() const { return static_cast<T*>(p); }
    void upload(const void* src, size_t n) {
        if (n > bytes)
            throw std::runtime_error("upload overflow");
        hip_check(hipMemcpy(p, src, n, hipMemcpyHostToDevice), "HtoD");
    }
    void download(void* dst, size_t n) const {
        hip_check(hipMemcpy(dst, p, n, hipMemcpyDeviceToHost), "DtoH");
    }
    void zero() { hip_check(hipMemset(p, 0, bytes), "memset"); }
};

inline std::vector<int> read_i32(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f)
        throw std::runtime_error("missing " + path);
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0 || n % 4)
        throw std::runtime_error("bad size " + path);
    std::vector<int> v(size_t(n) / 4);
    if (fread(v.data(), 1, size_t(n), f) != size_t(n))
        throw std::runtime_error("truncated " + path);
    fclose(f);
    return v;
}

inline std::vector<float> read_f32(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f)
        throw std::runtime_error("missing " + path);
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0 || n % 4)
        throw std::runtime_error("bad size " + path);
    std::vector<float> v(size_t(n) / 4);
    if (fread(v.data(), 1, size_t(n), f) != size_t(n))
        throw std::runtime_error("truncated " + path);
    fclose(f);
    return v;
}

inline float half_to_float(uint16_t h) {
    uint32_t s = (uint32_t(h) & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 1023u, b;
    if (e == 0) {
        if (!m)
            b = s;
        else {
            int sh = 0;
            while (!(m & 0x400u)) {
                m <<= 1;
                sh++;
            }
            m &= 0x3ffu;
            b = s | ((113u - sh) << 23) | (m << 13);
        }
    } else if (e == 31)
        b = s | 0x7f800000u | (m << 13);
    else
        b = s | ((e + 112u) << 23) | (m << 13);
    union {
        uint32_t u;
        float f;
    } conv{b};
    return conv.f;
}

inline std::vector<float> read_weights(const std::string& path) {
    try {
        return read_f32(path);
    } catch (...) {
        if (path.size() > 4 && path.substr(path.size() - 4) == ".f32") {
            auto half = path.substr(0, path.size() - 4) + ".f16";
            FILE* f = fopen(half.c_str(), "rb");
            if (!f)
                throw;
            fseek(f, 0, SEEK_END);
            long n = ftell(f);
            fseek(f, 0, SEEK_SET);
            if (n < 0 || n % 2) {
                fclose(f);
                throw std::runtime_error("bad half size " + half);
            }
            std::vector<uint16_t> h(size_t(n) / 2);
            if (fread(h.data(), 1, size_t(n), f) != size_t(n)) {
                fclose(f);
                throw std::runtime_error("truncated " + half);
            }
            fclose(f);
            std::vector<float> v(h.size());
            for (size_t i = 0; i < h.size(); i++)
                v[i] = half_to_float(h[i]);
            return v;
        }
        throw;
    }
}

inline DeviceBuf upload_tiled_e4m3(const float* src, size_t N, size_t K) {
    std::vector<u8> tiled(N * K);
    pack_tiled_e4m3(tiled.data(), src, N, K);
    DeviceBuf b(tiled.size());
    b.upload(tiled.data(), tiled.size());
    return b;
}

inline DeviceBuf upload_tiled_half(const float* src, size_t N, size_t K) {
    std::vector<__half> tiled(N * K);
    pack_tiled_half(tiled.data(), src, N, K);
    DeviceBuf b(tiled.size() * 2);
    b.upload(tiled.data(), tiled.size() * 2);
    return b;
}

// Weights for the four kernels that read the DLSS5_W16 format. Use this rather than picking a
// packer by hand: it is the single place the upload side consults the flag, which is what keeps it
// from disagreeing with the dispatch side. (Weights for other kernels -- stage.hip's split tiles,
// launch_linear_half's genuine f16 -- are not governed by the flag and keep their own uploads.)
inline DeviceBuf upload_tiled_weights(const float* src, size_t N, size_t K) {
    return dlss5_w16() ? upload_tiled_half(src, N, K) : upload_tiled_e4m3(src, N, K);
}

} // namespace dlss5
