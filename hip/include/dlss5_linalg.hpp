// Wave-matrix wrapper matching dx::linalg 16x32 A · 32x16 B → 16x16 C.
// gfx1201 native WMMA is 16x16x16 FP8; one D3D12 MMA is two gfx12 ops along K.
#pragma once
#include "dlss5_common.hpp"
#include <rocwmma/rocwmma.hpp>

namespace dlss5 {
namespace linalg {

using rocwmma::col_major;
using rocwmma::matrix_a;
using rocwmma::matrix_b;
using rocwmma::accumulator;
using rocwmma::row_major;
using rocwmma::float8_t;
using f16 = rocwmma::hfloat16_t;

template <typename DataT, typename Layout>
using FragA = rocwmma::fragment<matrix_a, 16, 16, 16, DataT, Layout>;
template <typename DataT, typename Layout>
using FragB = rocwmma::fragment<matrix_b, 16, 16, 16, DataT, Layout>;
using FragC = rocwmma::fragment<accumulator, 16, 16, 16, float>;

// RDNA 3 (gfx11): rocWMMA has no FP8 WMMA, only f16 operands with an f32
// accumulator (same 16x16x16 block, wave32). E4M3 codes (|v| <= 448, 3-bit
// mantissa) are exactly representable in f16, so FP8 matrices are decoded
// into f16 fragments at load time. Buffers, weight tiles and byte offsets stay
// E4M3, so the converter and host code are unchanged.
#if DLSS5_GFX11
template <typename DataT> struct FragStorage { using type = DataT; };
template <> struct FragStorage<float8_t> { using type = f16; };
template <typename DataT> using StorageT = typename FragStorage<DataT>::type;
template <typename DataT>
constexpr bool kDecodeE4M3 = rocwmma::is_same_v<DataT, float8_t>;

// Compact 16x16 f16 copy of a strided E4M3 region. row_major: element (r, c)
// = src[r * ldm + c]; col_major: element (r, c) = src[c * ldm + r]. Both are
// written so the compact buffer uses the same layout with ldm = 16.
__device__ inline void decode_e4m3_16x16(f16* dst, const u8* src, uint ldm) {
    for (uint i = 0; i < 16u; i++)
        for (uint j = 0; j < 16u; j++)
            dst[i * 16u + j] = f16(from_e4m3(src[i * ldm + j]));
}
template <typename Frag>
__device__ inline void load_e4m3(Frag& frag, const u8* src, uint ldm) {
    f16 tmp[256];
    decode_e4m3_16x16(tmp, src, ldm);
    rocwmma::load_matrix_sync(frag, static_cast<const f16*>(tmp), 16u);
}
#define DLSS5_LOAD_FRAG(frag, p, ldm)                                                  \
    do {                                                                               \
        if constexpr (kDecodeE4M3<DataT>)                                                      \
            load_e4m3(frag, reinterpret_cast<const u8*>(p), ldm);                      \
        else                                                                           \
            rocwmma::load_matrix_sync(frag, reinterpret_cast<const StorageT<DataT>*>(p), \
                                      ldm);                                            \
    } while (0)
#else
template <typename DataT> using StorageT = DataT;
#define DLSS5_LOAD_FRAG(frag, p, ldm) rocwmma::load_matrix_sync(frag, p, ldm)
#endif

// gfx12 f32 accumulator layout (GPUOpen RDNA 4): 8 elems/thread,
// column = lane%16, rows = (lane>=16 ? 8 : 0) + i.
__device__ inline uint2 acc_coord(uint i) {
    uint lane = threadIdx.x & 31u;
    return uint2{(lane >= 16u ? 8u : 0u) + i, lane & 15u};
}

template <typename DataT>
struct MatrixA {
    FragA<StorageT<DataT>, row_major> k0, k1;
    template <typename Ptr>
    __device__ static MatrixA Load(Ptr buf, uint byte_off, uint stride_bytes) {
        const DataT* p = reinterpret_cast<const DataT*>(
            reinterpret_cast<const char*>(buf) + byte_off);
        uint ldm = stride_bytes / uint(sizeof(DataT));
        MatrixA a;
        DLSS5_LOAD_FRAG(a.k0, p, ldm);
        DLSS5_LOAD_FRAG(a.k1, p + 16, ldm);
        return a;
    }
};

template <typename DataT>
struct MatrixB {
    FragB<StorageT<DataT>, row_major> k0r, k1r;
    FragB<StorageT<DataT>, col_major> k0c, k1c;
    bool row = false;
    template <typename Ptr>
    __device__ static MatrixB LoadRow(Ptr buf, uint byte_off, uint stride_bytes) {
        const DataT* p = reinterpret_cast<const DataT*>(
            reinterpret_cast<const char*>(buf) + byte_off);
        uint ldm = stride_bytes / uint(sizeof(DataT));
        MatrixB b;
        b.row = true;
        DLSS5_LOAD_FRAG(b.k0r, p, ldm);
        DLSS5_LOAD_FRAG(b.k1r, p + 16 * ldm, ldm);
        return b;
    }
    template <typename Ptr>
    __device__ static MatrixB LoadCol(Ptr buf, uint byte_off, uint stride_bytes) {
        const DataT* p = reinterpret_cast<const DataT*>(
            reinterpret_cast<const char*>(buf) + byte_off);
        uint ldm = stride_bytes / uint(sizeof(DataT));
        MatrixB b;
        b.row = false;
        DLSS5_LOAD_FRAG(b.k0c, p, ldm);
        DLSS5_LOAD_FRAG(b.k1c, p + 16, ldm);
        return b;
    }
};

struct MatrixC {
    FragC acc{};
    __device__ static MatrixC Splat(float v) {
        MatrixC c;
        rocwmma::fill_fragment(c.acc, v);
        return c;
    }
    __device__ uint Length() const { return FragC::num_elements; }
    __device__ float Get(uint i) const { return acc[i]; }
    __device__ void Set(uint i, float v) { acc[i] = v; }
    __device__ uint2 GetCoordinate(uint i) const { return acc_coord(i); }

    template <typename DataT>
    __device__ void MultiplyAccumulate(const MatrixA<DataT>& a, const MatrixB<DataT>& b) {
        if (b.row) {
            rocwmma::mma_sync(acc, a.k0, b.k0r, acc);
            rocwmma::mma_sync(acc, a.k1, b.k1r, acc);
        } else {
            rocwmma::mma_sync(acc, a.k0, b.k0c, acc);
            rocwmma::mma_sync(acc, a.k1, b.k1c, acc);
        }
    }

    template <typename Ptr>
    __device__ void StoreF32(Ptr buf, uint byte_off, uint stride_bytes) const {
        float* p = reinterpret_cast<float*>(reinterpret_cast<char*>(buf) + byte_off);
        rocwmma::store_matrix_sync(p, acc, stride_bytes / 4u, rocwmma::mem_row_major);
    }

    // Element-wise stores: gfx12 accumulator Cast<> is not a rocWMMA store type.
    template <typename Ptr>
    __device__ void StoreF8(Ptr buf, uint byte_off, uint stride_bytes) const {
        u8* p = reinterpret_cast<u8*>(reinterpret_cast<char*>(buf) + byte_off);
        for (uint i = 0; i < Length(); i++) {
            uint2 rc = GetCoordinate(i);
            p[rc.x * stride_bytes + rc.y] = e4m3_byte(Get(i));
        }
    }

    template <typename Ptr>
    __device__ void StoreF16(Ptr buf, uint byte_off, uint stride_bytes) const {
        f16* p = reinterpret_cast<f16*>(reinterpret_cast<char*>(buf) + byte_off);
        uint ldm = stride_bytes / 2u;
        for (uint i = 0; i < Length(); i++) {
            uint2 rc = GetCoordinate(i);
            p[rc.x * ldm + rc.y] = f16(Get(i));
        }
    }

    template <typename Ptr>
    __device__ static MatrixC LoadF32(Ptr buf, uint byte_off, uint stride_bytes) {
        const float* p = reinterpret_cast<const float*>(
            reinterpret_cast<const char*>(buf) + byte_off);
        MatrixC c;
        rocwmma::load_matrix_sync(c.acc, p, stride_bytes / 4u, rocwmma::mem_row_major);
        return c;
    }
};

template <typename DataT>
__device__ inline MatrixC Multiply(const MatrixA<DataT>& a, const MatrixB<DataT>& b) {
    MatrixC c = MatrixC::Splat(0.f);
    c.MultiplyAccumulate(a, b);
    return c;
}

using A8 = MatrixA<float8_t>;
using B8 = MatrixB<float8_t>;
using A16 = MatrixA<f16>;
using B16 = MatrixB<f16>;
using C32 = MatrixC;

} // namespace linalg
} // namespace dlss5
