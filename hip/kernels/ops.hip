// Logical FP32 storage keeps half-valued residuals distinct from FP8 operands.
#pragma once
#include "dlss5_linalg.hpp"

namespace dlss5 {

// mode: 0 half raw, 1 FP8(half), 2 activated FP8, 3 unrounded FP32,
// 4 C32 final H(branch + residual*scale) without H(product), 5 direct FP8.
// ordered: 0 FP32 accumulation, 1 H after each K32, 2 four independent
// FP32 K partitions combined with H(residual*scale) (fast ViT reduce).
template <bool F8IN, bool F8OUT = false>
__global__ void k_linear_f32(const float* input, const u8* input_f8, const u8* weights,
                             float* output, u8* output_f8, uint m, uint n, uint k,
                             const float* residual, const float* scales, int mode, int ordered,
                             bool matrix_residual = false) {
    using namespace linalg;
    const uint first = blockIdx.x * 16, col = blockIdx.y * 64 + (threadIdx.x / 32) * 16;
    __shared__ u8 tile[512];
    C32 acc = C32::Splat(0.f);
    if (residual && mode != 4)
        for (uint i = 0; i < acc.Length(); ++i) {
            uint2 rc = acc.GetCoordinate(i);
            if (first + rc.x < m && col < n)
                acc.Set(i, matrix_residual
                    ? residual[(first + rc.x) * n + col + rc.y] * scales[col + rc.y]
                    : H(residual[(first + rc.x) * n + col + rc.y] * scales[col + rc.y]));
        }
    C32 partition = C32::Splat(0.f);
    for (uint g = 0; g < k / 32; ++g) {
        {
            const uint col = (threadIdx.x % 8u) * 4u;
            const uint row = first + threadIdx.x / 8u;
            if (F8IN) {
                *reinterpret_cast<u32*>(tile + (threadIdx.x / 8u) * 32u + col) =
                    row < m ? *reinterpret_cast<const u32*>(input_f8 + row * k + g * 32u + col)
                            : 0u;
            } else {
                const float4 v = row < m
                    ? *reinterpret_cast<const float4*>(input + row * k + g * 32u + col)
                    : make_float4(0.f, 0.f, 0.f, 0.f);
                *reinterpret_cast<u32*>(tile + (threadIdx.x / 8u) * 32u + col) =
                    u32(e4m3_byte(v.x)) | u32(e4m3_byte(v.y)) << 8 |
                    u32(e4m3_byte(v.z)) << 16 | u32(e4m3_byte(v.w)) << 24;
            }
        }
        __syncthreads();
        A8 a = A8::Load(tile, 0, 32);
        // Four output waves share one converted input tile. In the final
        // partial group, inactive waves read a valid tile and omit stores.
        B8 b = B8::LoadRow(weights, (((col < n ? col : 0) / 16) * (k / 32) + g) * 512, 16);
        if (ordered == 2) {
            partition.MultiplyAccumulate(a,b);
            if ((g+1) % (k/128) == 0) {
                for(uint i=0;i<acc.Length();++i) acc.Set(i,acc.Get(i)+partition.Get(i));
                partition=C32::Splat(0.f);
            }
        } else if (ordered == 1) {
            C32 part = Multiply(a, b);
            for (uint i = 0; i < acc.Length(); ++i) acc.Set(i, H(acc.Get(i) + part.Get(i)));
        } else acc.MultiplyAccumulate(a, b);
        __syncthreads();
    }
    for (uint i = 0; i < acc.Length(); ++i) {
        uint2 rc = acc.GetCoordinate(i);
        float v = acc.Get(i);
        if (mode == 4 && residual && first + rc.x < m && col < n)
            v += residual[(first + rc.x) * n + col + rc.y] * scales[col + rc.y];
        u8 qout = 0;
        if (mode == 2) {
            u8 q = e4m3_byte(ActivatePoly(v));
            v = from_e4m3(q);
            if constexpr (F8OUT) qout = q;
        } else if (mode == 1) {
            u8 q = e4m3_byte(H(v));
            v = from_e4m3(q);
            if constexpr (F8OUT) qout = q;
        } else if (mode == 5) {
            u8 q = e4m3_byte(v);
            v = from_e4m3(q);
            if constexpr (F8OUT) qout = q;
        } else if (mode == 0 || mode == 4) v = H(v);
        if (first + rc.x < m && col < n) {
            output[(first + rc.x) * n + col + rc.y] = v;
            if constexpr (F8OUT)
                output_f8[(first + rc.x) * n + col + rc.y] = qout;
        }
    }
}

template <int C, bool F8OUT = false>
__global__ void k_ffn_f32(const float* input, const u8* weights, const float* scales,
                          float* output, u8* output_f8, uint tokens, bool chain_residual = false,
                          bool precise_c32 = false) {
    using namespace linalg;
    constexpr uint hidden_c = 4 * C;
    const uint first = blockIdx.x * 16, wave = threadIdx.x / 32;
    // Single LDS buffer: the input staging (16*C bytes) aliases the start of the
    // hidden buffer (16*hidden_c bytes). The input is only read during the expand
    // MMA loop and hidden is only written after it, so one barrier makes the
    // aliasing safe. Cuts LDS 16*C+16*4C -> 16*4C (e.g. C32 2560B->2048B).
    __shared__ u8 hidden[16 * hidden_c];
    u8* in = hidden;
    for (uint i = threadIdx.x; i < 16 * C; i += blockDim.x)
        in[i] = first + i / C < tokens ? e4m3_byte(input[first * C + i]) : 0;
    __syncthreads();
    C32 ex[4];
    for (uint n = 0; n < 4; ++n) ex[n] = C32::Splat(0.f);
    for (uint g = 0; g < C / 32; ++g) {
        A8 a = A8::Load(in, g * 32, C);
        for (uint n = 0; n < 4; ++n) {
            uint col = (wave * 4 + n) * 16;
            B8 b = B8::LoadRow(weights, ((col / 16) * (C / 32) + g) * 512, 16);
            ex[n].MultiplyAccumulate(a, b);
        }
    }
    __syncthreads();
    for (uint n = 0; n < 4; ++n) {
        for (uint i = 0; i < ex[n].Length(); ++i) ex[n].Set(i,
            C == 32 && precise_c32 ? ActivatePolyC32(ex[n].Get(i)) : ActivatePoly(ex[n].Get(i)));
        ex[n].StoreF8(hidden, (wave * 4 + n) * 16, hidden_c);
    }
    __syncthreads();
    C32 acc = C32::Splat(0.f);
    if constexpr (C == 32)
        for (uint i = 0; i < acc.Length(); ++i) {
            uint2 rc = acc.GetCoordinate(i);
            if (first + rc.x < tokens)
                acc.Set(i, (chain_residual
                    ? from_e4m3(e4m3_byte(input[(first + rc.x) * C + wave * 16 + rc.y]))
                    : input[(first + rc.x) * C + wave * 16 + rc.y]) * scales[wave * 16 + rc.y]);
        }
    for (uint g = 0; g < hidden_c / 32; ++g) {
        A8 a = A8::Load(hidden, g * 32, hidden_c);
        B8 b = B8::LoadRow(weights, hidden_c * C + (wave * (hidden_c / 32) + g) * 512, 16);
        acc.MultiplyAccumulate(a, b);
    }
    for (uint i = 0; i < acc.Length(); ++i) {
        uint2 rc = acc.GetCoordinate(i);
        if (first + rc.x < tokens) {
            if constexpr (C == 32) {
                output[(first + rc.x) * C + wave * 16 + rc.y] = H(acc.Get(i));
            } else {
                u8 q = e4m3_byte(acc.Get(i));
                if constexpr (F8OUT)
                    output_f8[(first + rc.x) * C + wave * 16 + rc.y] = q;
                output[(first + rc.x) * C + wave * 16 + rc.y] = from_e4m3(q);
            }
        }
    }
}

// One thread per 32-channel item: 8 aligned float4 loads, the original serial
// FP32 sum (a tree reduction changes FP8 rounding), packed 2x float4 stores.
__global__ void k_normalize_qkv(const float* input, const float* scales, u8* output,
                                uint tokens, uint c, float qgain, bool half_squares = true) {
    const uint count = tokens * (c / 32) * 3;
    const uint item = blockIdx.x * 256u + threadIdx.x;
    if (item >= count)
        return;
    const float4* row = reinterpret_cast<const float4*>(input + item * 32u);
    float4 v0 = row[0], v1 = row[1], v2 = row[2], v3 = row[3];
    float4 v4 = row[4], v5 = row[5], v6 = row[6], v7 = row[7];
    float vals[32];
    vals[0] = v0.x; vals[1] = v0.y; vals[2] = v0.z; vals[3] = v0.w;
    vals[4] = v1.x; vals[5] = v1.y; vals[6] = v1.z; vals[7] = v1.w;
    vals[8] = v2.x; vals[9] = v2.y; vals[10] = v2.z; vals[11] = v2.w;
    vals[12] = v3.x; vals[13] = v3.y; vals[14] = v3.z; vals[15] = v3.w;
    vals[16] = v4.x; vals[17] = v4.y; vals[18] = v4.z; vals[19] = v4.w;
    vals[20] = v5.x; vals[21] = v5.y; vals[22] = v5.z; vals[23] = v5.w;
    vals[24] = v6.x; vals[25] = v6.y; vals[26] = v6.z; vals[27] = v6.w;
    vals[28] = v7.x; vals[29] = v7.y; vals[30] = v7.z; vals[31] = v7.w;
    uint head = item % (c / 32), part = (item / (c / 32)) % 3;
    float sum = 0.f;
    if (part < 2)
        for (uint ch = 0; ch < 32; ++ch) {
            // Opaque register multiplies (sq_f32/sq_half_f32) block the FMA
            // contraction of sum += v*v without the volatile scratch round trip
            // that spilled 40 B/lane on gfx12.
            float square = half_squares ? sq_half_f32(vals[ch]) : sq_f32(vals[ch]);
            sum += square;
        }
    float inv = part == 2 ? 1.f : rsqrtf(fmaxf(sum, 6.198883056640625e-5f)) *
                (part == 0 ? scales[head] * qgain : 1.f);
    u8 b[32];
    for (uint ch = 0; ch < 32; ch++)
        b[ch] = e4m3_byte(vals[ch] * inv);
    u8* out = output + item * 32u;
    for (uint k = 0; k < 8; k++)
        *reinterpret_cast<u32*>(out + k * 4u) =
            u32(b[k * 4]) | u32(b[k * 4 + 1]) << 8 | u32(b[k * 4 + 2]) << 16 |
            u32(b[k * 4 + 3]) << 24;
}

// Four channels per thread: the 4 lanes are contiguous, so win/tok are shared
// and both ends are 16B vector accesses.
__global__ void k_reframe_f32(const float* src, float* dst, uint w, uint h, uint sw, uint sh,
                               uint px, uint py, uint c, int crop, int quantize) {
    uint i4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    if (i4 + 4u > (crop ? w * h : sw * sh) * c) return;
    uint ch = i4 % c, t = i4 / c;
    uint x = t % (crop ? w : sw), y = t / (crop ? w : sw);
    uint wx = crop ? x + px : x, wy = crop ? y + py : y;
    uint win = (wy / 8) * (sw / 8) + wx / 8, tok = (wy % 8) * 8 + wx % 8;
    float4 v;
    if (crop)
        v = *reinterpret_cast<const float4*>(src + (win * 64u + tok) * c + ch);
    else
        v = x >= px && y >= py && x < w + px && y < h + py
            ? *reinterpret_cast<const float4*>(src + ((y - py) * w + x - px) * c + ch)
            : make_float4(0.f, 0.f, 0.f, 0.f);
    if (quantize)
        v = float4(from_e4m3(e4m3_byte(v.x)), from_e4m3(e4m3_byte(v.y)),
                   from_e4m3(e4m3_byte(v.z)), from_e4m3(e4m3_byte(v.w)));
    if (crop)
        *reinterpret_cast<float4*>(dst + i4) = v;
    else
        *reinterpret_cast<float4*>(dst + (win * 64u + tok) * c + ch) = v;
}

__global__ void k_pool_f32(const float* src, float* dst, uint w, uint h, uint c, uint ow, uint oh) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ow * oh * c) return;
    uint ch = i % c, t = i / c, x = t % ow, y = t / ow;
    if (x >= w / 2 || y >= h / 2) { dst[i] = 0.f; return; }
    uint base = (y * 2 * w + x * 2) * c + ch;
    float top = H(src[base] + src[base + c]);
    float bottom = H(src[base + w * c] + src[base + (w + 1) * c]);
    dst[i] = from_e4m3(e4m3_byte(H(H(top + bottom) * .25f)));
}

// True half operands for ViT QKV, decoder entry and pool/head matrices.
// partitions reproduce the fast shader's independent K reductions; ordered
// rounds after each K32 only for the head/DS reference path.
__global__ void k_linear_half(const float* input,const __half* weights,float* output,
 uint m,uint n,uint k,uint partitions,int ordered,int raw) {
 using namespace linalg;
 uint first=blockIdx.x*16,col=blockIdx.y*16;
 __shared__ __half tile[512];
 C32 total=C32::Splat(0.f);
   for(uint part=0;part<partitions;++part){
   C32 acc=C32::Splat(0.f);
   for(uint g=part*(k/32/partitions);g<(part+1)*(k/32/partitions);++g){
    {
     const uint col=(threadIdx.x%8u)*4u;
     for(uint r=threadIdx.x/8u;r<16;r+=4){
      const uint row=first+r;
      const float4 v=row<m?*reinterpret_cast<const float4*>(input+row*k+g*32u+col):make_float4(0.f,0.f,0.f,0.f);
      const __half h0=__float2half(v.x),h1=__float2half(v.y),h2=__float2half(v.z),h3=__float2half(v.w);
      *reinterpret_cast<uint2*>(tile+r*32u+col)=
       uint2(u32(__half_as_ushort(h0))|u32(__half_as_ushort(h1))<<16,
             u32(__half_as_ushort(h2))|u32(__half_as_ushort(h3))<<16);
     }
    }
    __syncthreads();
   A16 a=A16::Load(tile,0,64);B16 b=B16::LoadRow(weights,((col/16)*(k/32)+g)*1024,32);
   if(ordered){C32 p=Multiply(a,b);for(uint i=0;i<acc.Length();++i)acc.Set(i,H(acc.Get(i)+p.Get(i)));}
   else acc.MultiplyAccumulate(a,b);
   __syncthreads();
  }
  for(uint i=0;i<total.Length();++i)total.Set(i,total.Get(i)+acc.Get(i));
 }
 for(uint i=0;i<total.Length();++i){uint2 rc=total.GetCoordinate(i);if(first+rc.x<m){float v=total.Get(i);output[(first+rc.x)*n+col+rc.y]=raw==2?v:raw==1?H(v):from_e4m3(e4m3_byte(H(v)));}}
}

// Split FFWD group parallel, logical window-major input/output. No opaque
// tiled activation format crosses a launch boundary.
template <bool F8OUT>
__global__ void k_split_f32(const float* in,const u8* pre,const u8* expand,const u8* contract,
                            float* out,u8* out_f8,uint tokens){
 using namespace linalg;
 uint first=blockIdx.x*16,group=blockIdx.y,wave=threadIdx.x/32;
 __shared__ u8 tile[512],mix[16*64],hid[16*256];
  C32 a=C32::Splat(0.f);
  for(uint g=0;g<16;++g){
   {
    const uint col=(threadIdx.x%8u)*4u, row=first+threadIdx.x/8u;
    const float4 v=row<tokens?*reinterpret_cast<const float4*>(in+row*512u+g*32u+col):make_float4(0.f,0.f,0.f,0.f);
    *reinterpret_cast<u32*>(tile+(threadIdx.x/8u)*32u+col)=
     u32(e4m3_byte(v.x))|u32(e4m3_byte(v.y))<<8|u32(e4m3_byte(v.z))<<16|u32(e4m3_byte(v.w))<<24;
   }
   __syncthreads();
  a.MultiplyAccumulate(A8::Load(tile,0,32),B8::LoadRow(pre,((group*4+wave)*16+g)*512,16));__syncthreads();
 }
 for(uint i=0;i<a.Length();++i)a.Set(i,H(a.Get(i)));a.StoreF8(mix,wave*16,64);__syncthreads();
 for(uint j=0;j<4;++j){C32 acc=C32::Splat(0.f);uint col=wave*64+j*16;
  for(uint g=0;g<2;++g)acc.MultiplyAccumulate(A8::Load(mix,g*32,64),B8::LoadRow(expand,group*16384+((col/16)*2+g)*512,16));
  for(uint i=0;i<acc.Length();++i)acc.Set(i,ActivatePoly(H(acc.Get(i))));acc.StoreF8(hid,col,256);
 }
 __syncthreads();C32 acc=C32::Splat(0.f);
 for(uint g=0;g<8;++g)acc.MultiplyAccumulate(A8::Load(hid,g*32,256),B8::LoadRow(contract,group*16384+(wave*8+g)*512,16));
  for(uint i=0;i<acc.Length();++i){uint2 rc=acc.GetCoordinate(i);if(first+rc.x<tokens){u8 q=e4m3_byte(H(acc.Get(i)));if constexpr(F8OUT)out_f8[(first+rc.x)*512+group*64+wave*16+rc.y]=q;out[(first+rc.x)*512+group*64+wave*16+rc.y]=from_e4m3(q);}}
}

__global__ void k_rgb_graph(const float* in,float* raster,float* tiles,uint w,uint h,uint ow,uint oh){
 uint p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=ow*oh)return;
 uint x=p%ow,y=p/ow,rx=x%(2*(w-1)),ry=y%(2*(h-1));rx=rx<w?rx:2*(w-1)-rx;ry=ry<h?ry:2*(h-1)-ry;
 uint t=((y/8)*(ow/8)+x/8)*64+(y%8)*8+x%8;
 for(uint c=0;c<4;++c){float v=in[(ry*w+rx)*4+c];raster[p*4+c]=v;tiles[t*4+c]=v;}
}
__global__ void k_gather_f32(const float* in,const int* map,float* out,uint n){uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=in[map[i]];}
__global__ void k_up_f32(const float* lo,const float* skip,const float* scale,float* out,uint iw,uint ow,uint oh,uint c){
 uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ow*oh*c)return;
 uint ch=i%c,p=i/c,x=p%ow,y=p/ow;
 float v=H(lo[((y/2)*iw+x/2)*c+ch]+from_e4m3(e4m3_byte(skip[i]))*scale[ch]);
 out[i]=c==32?v:from_e4m3(e4m3_byte(v));
}
__global__ void k_post_merge_f32(const float* low,const float* pre,const float* scales,float* out,uint w,uint h,bool main8_low = false){
 uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=w*h*32)return;
 uint ch=i%32,p=i/32,lp=((p/w)/2)*(w/2)+(p%w)/2;
 float lo=main8_low?from_e4m3(e4m3_byte(low[lp*32+ch])):low[lp*32+ch];
 out[i]=H(H(lo*scales[ch])+from_e4m3(e4m3_byte(pre[i]))*scales[32+ch]);
}
__global__ void k_head_f32(const float* feat,const float* weights,const float* color,float* out,uint n){
 uint p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;
 for(uint row=0;row<3;++row){float acc=0;for(uint c=0;c<32;++c)acc+=feat[p*32+c]*weights[row*32+c];
  float base=color[p*4+row]*.125f-.0625f;float encoded=H(acc)*.03125f+base;out[p*3+row]=fminf(fmaxf(encoded*8.f+.5f,0.f),1.f);}
}
// Per-lane trace reductions avoid atomics and CPU synchronization in the graph.
__global__ void k_trace_f32(const float* src,uint n,float* stats,uint* bits){
 uint lane=threadIdx.x;float lo=INFINITY,hi=-INFINITY,sum=0;uint bad=0,hash=0;
 for(uint i=lane;i<n;i+=256){float v=src[i];if(!isfinite(v))++bad;else{lo=fminf(lo,v);hi=fmaxf(hi,v);sum+=v;}uint z=__float_as_uint(v)^((i+1)*0x9e3779b9u);z=(z^(z>>16))*0x85ebca6bu;z=(z^(z>>13))*0xc2b2ae35u;hash^=z^(z>>16);}
 stats[lane*3]=lo;stats[lane*3+1]=hi;stats[lane*3+2]=sum;bits[lane*2]=bad;bits[lane*2+1]=hash;
}

} // namespace dlss5
