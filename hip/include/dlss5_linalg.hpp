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
// straight into f16 fragments at load time. Buffers, weight tiles and byte offsets stay
// E4M3, so the converter and host code are unchanged.
#if DLSS5_GFX11
template <typename DataT> struct FragStorage { using type = DataT; };
template <> struct FragStorage<float8_t> { using type = f16; };
template <typename DataT> using StorageT = typename FragStorage<DataT>::type;
template <typename DataT>
constexpr bool kDecodeE4M3 = rocwmma::is_same_v<DataT, float8_t>;

// e4m3_to_half(): dlss5_common.hpp

// gfx11 fragment element map (measured, tests/probe_frag.hip): 8 elements per lane.
// matrix_a: lane l holds row (l & 15), columns base..base+7 with base = l >= 16 ? 8 : 0.
// matrix_b (row_major and col_major fragments): lane l holds column (l & 15), rows base+i.
enum FragKind { kFragA, kFragBRow, kFragBCol };
template <typename Frag>
__device__ inline void load_e4m3(Frag& f, const u8* p, uint ldm, FragKind kind) {
    const uint l = threadIdx.x & 31u, lead = l & 15u, base = l >= 16u ? 8u : 0u;
    if (kind == kFragBRow) {
        // memory element (r, c) = p[r * ldm + c]; r = base + i, c = lead
        for (uint i = 0; i < 8u; i++) f[i] = e4m3_to_half(p[(base + i) * ldm + lead]);
    } else {
        // A row_major: p[row * ldm + col], row = lead;  B col_major: p[col * ldm + row], col = lead
        //
        // These eight bytes are ADJACENT, so they are read as two dwords rather than eight byte loads.
        // RDNA3 has no FP8 matrix instruction, so every one of these fragments is filled by hand, and
        // this branch serves every activation load in the network plus attention's K and V -- which are
        // activations, not weights, and so were untouched by the f16 weight conversion.
        //
        // Alignment holds wherever it is used: base is 0 or 8 and every ldm in this codebase is a
        // multiple of four, so all lanes agree with the tile base. The check is still there because a
        // misaligned dword load is a fault rather than a slowdown, and a future caller with an odd
        // stride should get the slow path, not a crash.
        const u8* q = p + lead * ldm + base;

        if ((reinterpret_cast<uintptr_t>(q) & 3u) == 0u) {
            const u32 lo = reinterpret_cast<const u32*>(q)[0];
            const u32 hi = reinterpret_cast<const u32*>(q)[1];

            f[0] = e4m3_to_half(lo & 0xffu);
            f[1] = e4m3_to_half((lo >> 8) & 0xffu);
            f[2] = e4m3_to_half((lo >> 16) & 0xffu);
            f[3] = e4m3_to_half(lo >> 24);
            f[4] = e4m3_to_half(hi & 0xffu);
            f[5] = e4m3_to_half((hi >> 8) & 0xffu);
            f[6] = e4m3_to_half((hi >> 16) & 0xffu);
            f[7] = e4m3_to_half(hi >> 24);
        } else {
            for (uint i = 0; i < 8u; i++) f[i] = e4m3_to_half(q[i]);
        }
    }
}
#define DLSS5_LOAD_FRAG(frag, p, ldm, kind)                                              \
    do {                                                                                 \
        if constexpr (kDecodeE4M3<DataT>)                                                \
            load_e4m3(frag, reinterpret_cast<const u8*>(p), ldm, kind);                  \
        else                                                                             \
            rocwmma::load_matrix_sync(frag, reinterpret_cast<const StorageT<DataT>*>(p), \
                                      ldm);                                              \
    } while (0)
#else
template <typename DataT> using StorageT = DataT;
#define DLSS5_LOAD_FRAG(frag, p, ldm, kind) rocwmma::load_matrix_sync(frag, p, ldm)
#endif

#if DLSS5_GFX11
// gfx11 f32 accumulator layout (measured on RX 7900 XT, tests/probe_gfx11.hip):
// 8 elems/thread, column = lane%16, rows interleaved: even rows on lanes 0..15,
// odd rows on lanes 16..31 (row = 2i + (lane>=16)).
__device__ inline uint2 acc_coord(uint i) {
    uint lane = threadIdx.x & 31u;
    return uint2{2u * i + (lane >= 16u ? 1u : 0u), lane & 15u};
}
#else
// gfx12 f32 accumulator layout (GPUOpen RDNA 4): 8 elems/thread,
// column = lane%16, rows = (lane>=16 ? 8 : 0) + i.
__device__ inline uint2 acc_coord(uint i) {
    uint lane = threadIdx.x & 31u;
    return uint2{(lane >= 16u ? 8u : 0u) + i, lane & 15u};
}
#endif

template <typename DataT>
struct MatrixA {
    FragA<StorageT<DataT>, row_major> k0, k1;
    template <typename Ptr>
    __device__ static MatrixA Load(Ptr buf, uint byte_off, uint stride_bytes) {
        const DataT* p = reinterpret_cast<const DataT*>(
            reinterpret_cast<const char*>(buf) + byte_off);
        uint ldm = stride_bytes / uint(sizeof(DataT));
        MatrixA a;
        DLSS5_LOAD_FRAG(a.k0, p, ldm, kFragA);
        DLSS5_LOAD_FRAG(a.k1, p + 16, ldm, kFragA);
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
        DLSS5_LOAD_FRAG(b.k0r, p, ldm, kFragBRow);
        DLSS5_LOAD_FRAG(b.k1r, p + 16 * ldm, ldm, kFragBRow);
        return b;
    }
    template <typename Ptr>
    __device__ static MatrixB LoadCol(Ptr buf, uint byte_off, uint stride_bytes) {
        const DataT* p = reinterpret_cast<const DataT*>(
            reinterpret_cast<const char*>(buf) + byte_off);
        uint ldm = stride_bytes / uint(sizeof(DataT));
        MatrixB b;
        b.row = false;
        DLSS5_LOAD_FRAG(b.k0c, p, ldm, kFragBCol);
        DLSS5_LOAD_FRAG(b.k1c, p + 16, ldm, kFragBCol);
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

    // Independent types for A and B. Both sides store f16 fragments whatever their source format --
    // FragStorage<float8_t>::type is f16 -- so mma_sync sees the same thing either way, and the only
    // thing that ever required them to match was deducing one DataT from two arguments. Splitting them
    // lets an E4M3 activation multiply an f16 weight, which is what measuring the E4M3 load costs
    // requires.
    template <typename TA, typename TB>
    __device__ void MultiplyAccumulate(const MatrixA<TA>& a, const MatrixB<TB>& b) {
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

    // f16 of the E4M3-ROUNDED value, which is what keeps this bit-identical.
    //
    // StoreF16 would write the accumulator's own f16, which is more accurate than the E4M3 the hidden
    // buffer holds today and therefore a different picture. Every E4M3 code is exactly representable in
    // f16, so rounding first and widening second reproduces exactly what load_e4m3 used to hand the
    // matrix unit -- one conversion here instead of ninety-six VALU ops on every fragment fill.
    template <typename Ptr>
    __device__ void StoreQuantF16(Ptr buf, uint byte_off, uint stride_bytes) const {
        f16* p = reinterpret_cast<f16*>(reinterpret_cast<char*>(buf) + byte_off);
        uint ldm = stride_bytes / 2u;

        for (uint i = 0; i < Length(); i++) {
            uint2 rc = GetCoordinate(i);
#if DLSS5_GFX11
            p[rc.x * ldm + rc.y] = e4m3_to_half(e4m3_byte(Get(i)));
#else
            // e4m3_to_half is a gfx11-only device helper: that architecture has no hardware E4M3
            // conversion, so the software one sits under #if DLSS5_GFX11. This branch exists only so the
            // HOST pass compiles -- HIP parses every __global__ body and everything it mentions on the
            // host to build its launch stub, and nothing here ever runs there. It does not have to be
            // right; it has to exist.
            p[rc.x * ldm + rc.y] = __float2half(Get(i));
#endif
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

// The A fragment for an f16 hidden buffer, named by STORAGE type for the same reason BW16 is: in the
// host pass StorageT is the identity, so MatrixA<f16> there would be a f16 fragment against B8's fp8 one
// and rocWMMA's "Input datatypes must be same size" would fire. On device it resolves to f16 and loads
// with a plain load_matrix_sync; in the host pass it resolves to float8_t, matches B, compiles, and never
// runs. This is the second time that trap has cost a build -- see BW16.
template <bool F16> struct HiddenA { using type = MatrixA<float8_t>; };
template <> struct HiddenA<true> { using type = MatrixA<StorageT<float8_t>>; };
using B16 = MatrixB<f16>;

// The weight fragment for the W16 path: whatever storage type THIS compilation pass gives an E4M3
// matrix, so matrix A and matrix B always agree on element size.
//
// On gfx11 that is f16, and loading it is a single vectorised load_matrix_sync -- which is exactly what
// the W16 experiment exists to measure against load_e4m3's eight strided single-byte reads. In the HOST
// pass, where __gfx1100__ is not defined and StorageT is the identity, it is float8_t instead: never
// executed, but it has to compile, because HIP instantiates every __global__ template on the host to
// build its launch stub. Naming B16 directly here fails that pass with rocWMMA's "Input datatypes must
// be same size", which is correct -- fp8 and f16 are not the same size, and only the device pass agrees
// that an fp8 matrix is stored as f16.
using BW16 = MatrixB<StorageT<float8_t>>;
using C32 = MatrixC;

} // namespace linalg
} // namespace dlss5
