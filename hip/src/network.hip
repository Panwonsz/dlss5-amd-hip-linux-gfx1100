#include "dlss5_network.hpp"
#include "dlss5_launch.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>

namespace dlss5 {
namespace {
constexpr uint W=1920,Ht=1152,N=W*Ht;
DeviceBuf floats(const float* p,size_t n){DeviceBuf d(n*4);d.upload(p,n*4);return d;}
// No catch-all fallback: an existing corrupt f32 is always an error.
std::vector<float> load(const std::string& dir,const std::string& name,size_t count){
 namespace fs=std::filesystem;fs::path p=fs::path(dir)/(name+".f32");bool half=false;
 if(!fs::exists(p)){p=fs::path(dir)/(name+".f16");half=true;}
 std::ifstream f(p,std::ios::binary|std::ios::ate);
 if(!f||f.tellg()!=std::streamoff(count*(half?2:4)))throw std::runtime_error("exact table size: "+p.string());
 f.seekg(0);std::vector<float> v(count);
 if(half){std::vector<uint16_t> h(count);f.read((char*)h.data(),count*2);for(size_t i=0;i<count;++i)v[i]=half_to_float(h[i]);}
 else f.read((char*)v.data(),count*4);
 if(!f)throw std::runtime_error("read table: "+p.string());
 for(float x:v)if(!std::isfinite(x))throw std::runtime_error("nonfinite table: "+p.string());
 return v;
}
DeviceBuf half_matrix(const float* v,uint n,uint k){
 std::vector<__half> t(size_t(n)*k);
 for(uint r=0;r<n;++r)for(uint c=0;c<k;++c)t[((r/16)*(k/32)+c/32)*512+(c%32)*16+r%16]=__float2half(v[size_t(r)*k+c]);
 DeviceBuf b(t.size()*2);b.upload(t.data(),t.size()*2);return b;
}
std::vector<int> map_read(const std::string&p){
 std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f||f.tellg()!=655360*4)throw std::runtime_error("map exact size: "+p);
 std::vector<int> v(655360);f.seekg(0);f.read((char*)v.data(),v.size()*4);if(!f)throw std::runtime_error("map read: "+p);
 std::vector<bool> seen(v.size());for(int x:v){if(x<0||size_t(x)>=v.size()||seen[x])throw std::runtime_error("map range/bijection: "+p);seen[x]=true;}return v;
}
struct StreamScope {hipStream_t old;explicit StreamScope(hipStream_t s):old(get_launch_stream()){set_launch_stream(s);}~StreamScope(){set_launch_stream(old);}};
struct Scratch {
  DeviceBuf a,b,bh,ah,d,q,q8,att8,hidden,hidden8,d8,f8,out8,a8,out16,a16;
  void create(){const size_t n=size_t(W+8)*(Ht+8)*32;a.alloc(n*4);b.alloc(n*4);bh.alloc(n*2);ah.alloc(n*2);d.alloc(n*4);q.alloc(n*12);q8.alloc(n*3);att8.alloc(n);hidden.alloc(640ull*4096*4);hidden8.alloc(640ull*4096);d8.alloc(n);f8.alloc(n);out8.alloc(n);a8.alloc(n);out16.alloc(n*2);a16.alloc(n*2);}
};
struct Block {
 uint c{},w{},h{},px{},py{},sw{},sh{};bool raw{},chain{};
 DeviceBuf fw,fs,ex,ct,p0,s0,qw,qs,bias,p1,s1;
 void create(const std::string&dir,int id,uint channels,uint width,uint height,uint shift,bool raw_output){
  c=channels;w=width;h=height;px=(shift&1)?4:0;py=(shift&2)?4:0;sw=(w+px+7)&~7u;sh=(h+py+7)&~7u;raw=raw_output;
  chain=c==32&&(id==2||id==3||id==4||id==67||id==68||id==69);
  std::string p=id==70?"post70":"block"+std::to_string(id);size_t m=size_t(c)*c;
  if(c==512){auto f=load(dir,p+"-ffwd",524288);fw=upload_tiled_e4m3(f.data(),512,512);
   std::vector<u8> e(131072),t(131072);for(uint g=0;g<8;++g){pack_tiled_e4m3(e.data()+g*16384,f.data()+262144+g*16384,256,64);pack_tiled_e4m3(t.data()+g*16384,f.data()+393216+g*16384,64,256);}ex.alloc(e.size());ex.upload(e.data(),e.size());ct.alloc(t.size());ct.upload(t.data(),t.size());
   auto fproj=load(dir,p+"-ffwd-projection",262656);p0=upload_tiled_e4m3(fproj.data(),c,c);s0=floats(fproj.data()+m,c);
  }else {auto f=load(dir,p+"-ffn",c==32?8736:9*m+c);size_t off=c==32?512:0;
   std::vector<u8> t(8*m);pack_tiled_e4m3(t.data(),f.data()+off,4*c,c);pack_tiled_e4m3(t.data()+4*m,f.data()+off+4*m,c,4*c);fw.alloc(t.size());fw.upload(t.data(),t.size());
   if(c==32)fs=floats(f.data()+8704,32);else{p0=upload_tiled_e4m3(f.data()+8*m,c,c);s0=floats(f.data()+9*m,c);}
  }
  auto a=load(dir,p+"-attention",4*m+(c/32)*4096+c/32+c);
  qw=upload_tiled_e4m3(a.data(),3*c,c);p1=upload_tiled_e4m3(a.data()+3*m,c,c);bias=floats(a.data()+4*m,(c/32)*4096);qs=floats(a.data()+4*m+(c/32)*4096,c/32);s1=floats(a.data()+4*m+(c/32)*4096+c/32,c);
 }
 // The only physical activation format here is window-major [window][64][C].
 // Residuals remain distinct from quantized matrix operands.
  void window(const float* in,float* out,Scratch&s,const u8* in_f8=nullptr,bool emit_f8=false,
              const __half* in_h=nullptr,bool emit_h=false){
   uint n=sw*sh;float* f=s.b.as<float>();
   if(c==512){
    if(in_f8)launch_split_f32_f8in_f8(in_f8,fw.as<u8>(),ex.as<u8>(),ct.as<u8>(),s.d.as<float>(),s.d8.as<u8>(),n);
    else launch_split_f32_f8(in,fw.as<u8>(),ex.as<u8>(),ct.as<u8>(),s.d.as<float>(),s.d8.as<u8>(),n);
   }
   else if(c==32){
    if(in_h)launch_ffn_f32_hin_hout(32,in_h,fw.as<u8>(),fs.as<float>(),s.bh.as<__half>(),n,chain,true);
    else launch_ffn_f32_hout(32,in,fw.as<u8>(),fs.as<float>(),s.bh.as<__half>(),n,chain,true);
   }
   else if(in_f8)launch_ffn_f32_f8in_f8out(c,in_f8,fw.as<u8>(),nullptr,s.d.as<float>(),s.d8.as<u8>(),n,chain,false);
   else launch_ffn_f32_f8(c,in,fw.as<u8>(),nullptr,s.d.as<float>(),s.d8.as<u8>(),n,chain,false);
   // mapfeature FFN projection: H(scale product); direct store only C64..256.
   // Non-C32 activations are already on the E4M3 grid, so p0/QKV consume the bytes
   // directly while the f32 copy stays available for residuals.
    if(c!=32){
     if(in_f8)launch_linear_f32_f8in_f8out_f8res(s.d8.as<u8>(),p0.as<u8>(),f,s.f8.as<u8>(),n,c,c,in_f8,s0.as<float>(),c==512?1:5,0,false);
     else launch_linear_f32_f8in_f8out(s.d8.as<u8>(),p0.as<u8>(),f,s.f8.as<u8>(),n,c,c,in,s0.as<float>(),c==512?1:5,0);
    }
   // QKV GEMM + per-head scaling + E4M3 in one launch (no f32 round trip).
   // C32 keeps the FFN output in f16: H(acc) is exactly representable, so the
   // f16 -> f32 -> E4M3 path is bit-identical with half the global traffic.
   if(c==32)launch_qkv_norm_f32_hin(32,s.bh.as<__half>(),qw.as<u8>(),s.q8.as<u8>(),n,qs.as<float>(),1.f,true);
   else launch_qkv_norm_f32_f8in(c,s.f8.as<u8>(),qw.as<u8>(),s.q8.as<u8>(),n,qs.as<float>(),1.f,c==32);
   launch_attention(c,s.q8.as<u8>(),bias.as<float>(),s.att8.as<u8>(),sw,sh,1,true);
    // mapoutput C64..256 uses NATIVE_MATRIX_RESIDUAL even for raw outputs.
    // C512 split shaders retain H(product) and H before FP8 stores.
    // The attention output is already E4M3; e4m3_byte(from_e4m3(x)) is the identity,
    // so the projection consumes the bytes directly (no f8->f32 round trip).
    if(c==32){
     if(emit_h)launch_linear_f32_f8in_hrid_hout_raster(s.att8.as<u8>(),p1.as<u8>(),s.ah.as<__half>(),s.out16.as<__half>(),n,c,c,s.bh.as<__half>(),s1.as<float>(),4,0,false,w,h,sw,px,py);
     else launch_linear_f32_f8in_hrid_hout(s.att8.as<u8>(),p1.as<u8>(),s.ah.as<__half>(),n,c,c,s.bh.as<__half>(),s1.as<float>(),4,0,false);
    }
    else if(emit_f8)launch_linear_f32_f8in_f8out_raster(s.att8.as<u8>(),p1.as<u8>(),out,s.out8.as<u8>(),n,c,c,f,s1.as<float>(),raw?0:c==512?1:5,0,c!=32&&c!=512,w,h,sw,px,py);
    else launch_linear_f32_f8in(s.att8.as<u8>(),p1.as<u8>(),out,n,c,c,f,s1.as<float>(),raw?0:c==512?1:5,0,c!=32&&c!=512);
  }
  void run(const float* in,float* out,Scratch&s,const u8* in_f8=nullptr,bool emit_f8=false,
           const __half* in_h=nullptr,bool emit_h=false){
   bool use_f8 = c!=32 && in_f8;
   bool use_h = c==32 && in_h;
   if(use_h) launch_reframe_hin_hout(in_h,s.a16.as<__half>(),w,h,sw,sh,px,py,c);
   else if(use_f8) launch_reframe_f8in_f8out(in_f8,s.a8.as<u8>(),w,h,sw,sh,px,py,c);
   else launch_reframe_f32(in,s.a.as<float>(),w,h,sw,sh,px,py,c,0,0);
   window(use_h||use_f8?in:s.a.as<float>(),s.a.as<float>(),s,use_f8?s.a8.as<u8>():nullptr,emit_f8,
          use_h?s.a16.as<__half>():nullptr,emit_h);
   // C32 p1 writes its H(v) output in f16; the crop reframe reads half the
   // global traffic and still emits the identical f32 raster.
   if(c==32)launch_reframe_f32_hin(s.ah.as<__half>(),out,w,h,sw,sh,px,py,c,1,0);
   else launch_reframe_f32(s.a.as<float>(),out,w,h,sw,sh,px,py,c,1,0);
  }
};
struct Vit {
 DeviceBuf ex,ct,cs,qw,qs,pw,ps;
 void create(const std::string&dir,int id){std::string p="block"+std::to_string(id);
  auto e=load(dir,p+"-expand",4194304),c=load(dir,p+"-contract",4195328),q=load(dir,p+"-qkv",3145760),o=load(dir,p+"-projection",1049600);
  ex=upload_tiled_e4m3(e.data(),4096,1024);ct=upload_tiled_e4m3(c.data(),1024,4096);cs=floats(c.data()+4194304,1024);
  qw=half_matrix(q.data(),3072,1024);qs=floats(q.data()+3145728,32);pw=upload_tiled_e4m3(o.data(),1024,1024);ps=floats(o.data()+1048576,1024);
 }
  void run(const float* in,float* out,Scratch&s){
   // The expand output is already on the E4M3 grid; keeping the bytes lets the
   // contract GEMM consume them directly (e4m3_byte(from_e4m3(q)) == q).
   launch_linear_f32_f8out(in,ex.as<u8>(),s.hidden.as<float>(),s.hidden8.as<u8>(),640,4096,1024,nullptr,nullptr,2,0);
   launch_linear_f32_f8in(s.hidden8.as<u8>(),ct.as<u8>(),s.b.as<float>(),640,1024,4096,in,cs.as<float>(),1,2);
  // QKV fast fused uses genuine F16 weights, raw FP32 dot before normalization.
  launch_linear_half(s.b.as<float>(),qw.as<__half>(),s.q.as<float>(),640,3072,1024,1,0,2);
  launch_normalize_qkv(s.q.as<float>(),qs.as<float>(),s.q8.as<u8>(),640,1024,5.65625f);
   // The normalized QKV is already on the E4M3 grid, so attention can consume
   // the bytes directly; the f8 -> f32 round trip and re-quantize are redundant.
   launch_vit_attention_f8(s.q8.as<u8>(),s.att8.as<u8>(),640,1024);
   launch_linear_f32_f8in(s.att8.as<u8>(),pw.as<u8>(),out,640,1024,1024,s.b.as<float>(),ps.as<float>(),1,2);
 }
};
struct Up {
 uint iw,ih,ow,oh,cin,cout;DeviceBuf wt,sc;
 void create(const std::string&dir,const std::string&name,uint w,uint h,uint width,uint height,uint ic,uint oc){iw=w;ih=h;ow=width;oh=height;cin=ic;cout=oc;auto v=load(dir,name,size_t(ic)*oc+oc);wt=half_matrix(v.data(),oc,ic);sc=floats(v.data()+size_t(ic)*oc,oc);}
 void run(const float* in,const float* skip,float* out,Scratch&s){launch_linear_half(in,wt.as<__half>(),s.d.as<float>(),iw*ih,cout,cin,cin==1024?4:1,0,1);launch_up_f32(s.d.as<float>(),skip,sc.as<float>(),out,iw,ih,ow,oh,cout);}
};
}
struct Network70::Impl {
 std::mutex mutex;bool ready=false;float gpu_ms=0;hipStream_t stream{};hipEvent_t ev0{},ev1{};
 DeviceBuf rgb,color,tiles,history,prefix_w,head_w,post_sc,head_rgb,map,inv;
 DeviceBuf raster[2],skip[6],out,trace_stats,trace_bits,ds[4];
 std::array<Block,71> blocks;Vit vit[8];Up up[5];Scratch s;
 std::array<uint,71> counts{};std::array<Trace,71> trace{};
 ~Impl(){if(stream)(void)hipStreamSynchronize(stream);if(ev0)(void)hipEventDestroy(ev0);if(ev1)(void)hipEventDestroy(ev1);if(stream)(void)hipStreamDestroy(stream);}
 void create(const std::string&dir){
  // Validate and upload before activating this instance; caller swaps only on success.
  auto fwd=map_read(dir+"/hwc-to-vit.i32"),back=map_read(dir+"/vit-to-hwc.i32");
  for(size_t i=0;i<fwd.size();++i)if(back[fwd[i]]!=int(i))throw std::runtime_error("ViT maps not inverse");
  map.alloc(fwd.size()*4);map.upload(fwd.data(),fwd.size()*4);inv.alloc(back.size()*4);inv.upload(back.data(),back.size()*4);
  const uint shift[]={0,3,1,2,0,3,1,2};
  blocks[0].create(dir,0,32,W,Ht,0,true);
  auto group=[&](int first,int count,uint c,uint w,uint h,bool enc){for(int i=0;i<count;++i)blocks[first+i].create(dir,first+i,c,w,h,shift[i],c==32||(enc&&i==count-1));};
  group(1,4,32,960,576,true);group(5,4,64,480,288,true);group(9,6,128,240,144,true);group(15,8,256,120,72,true);group(23,8,512,60,36,true);
  group(40,8,512,60,36,false);group(48,8,256,120,72,false);
  for(int i=0;i<6;++i)blocks[56+i].create(dir,56+i,128,240,144,shift[(i+2)%4],false);
  group(62,4,64,480,288,false);group(66,4,32,960,576,false);blocks[70].create(dir,70,32,W,Ht,3,true);
  for(int i=0;i<8;++i)vit[i].create(dir,31+i);
  auto prefix=load(dir,"block0-ffn",8736);prefix_w=floats(prefix.data(),512);
  int ids[]={4,8,14,22};for(int i=0;i<4;++i){uint c=32u<<i;auto v=load(dir,"block"+std::to_string(ids[i])+"-ds",2ull*c*c);ds[i]=half_matrix(v.data(),2*c,c);}
  auto hm=load(dir,"head-matrix",524288);head_w=half_matrix(hm.data(),1024,512);
  auto sc=load(dir,"post70-scales",64),hw=load(dir,"post70-head",96);post_sc=floats(sc.data(),64);head_rgb=floats(hw.data(),96);
  up[0].create(dir,"decoder39-weights",32,20,60,36,1024,512);up[1].create(dir,"block48-weights",60,36,120,72,512,256);up[2].create(dir,"block56-weights",120,72,240,144,256,128);up[3].create(dir,"block62-weights",240,144,480,288,128,64);up[4].create(dir,"block66-weights",480,288,960,576,64,32);
  rgb.alloc(1920ull*1080*16);color.alloc(size_t(N)*16);tiles.alloc(size_t(N)*16);out.alloc(size_t(N)*12);
  for(auto&r:raster)r.alloc(size_t(N)*32*4);
  for(int i=0;i<6;++i)skip[i].alloc((size_t(N)*32>>i)*4);
  s.create();trace_stats.alloc(71*256*3*4);trace_bits.alloc(71*256*2*4);
  hip_check(hipStreamCreate(&stream),"network stream");hip_check(hipEventCreate(&ev0),"network event0");hip_check(hipEventCreate(&ev1),"network event1");ready=true;
 }
 void run(const float* input,float* output,uint seed,const float* host_history){
  if(!ready||!input||!output)throw std::runtime_error("network not ready/null buffers");
  StreamScope scope(stream);bool tracing=std::getenv("DLSS5_HIP_TRACE")&&std::string(std::getenv("DLSS5_HIP_TRACE"))!="0";
  if(host_history){
   if(!history.p)history.alloc(size_t(N)*16);
   hip_check(hipMemcpyAsync(history.p,host_history,size_t(N)*16,hipMemcpyHostToDevice,stream),"history upload");
  }
  counts.fill(0);for(auto&t:trace)t=Trace{};
  auto mark=[&](int id,const float* data,uint count){++counts[id];trace[id].elements=count;if(tracing)launch_trace_f32(data,count,trace_stats.as<float>()+id*768,trace_bits.as<uint>()+id*512);};
  hip_check(hipMemcpyAsync(rgb.p,input,1920ull*1080*16,hipMemcpyHostToDevice,stream),"input upload");hip_check(hipEventRecord(ev0,stream),"start");
  launch_rgb_graph(rgb.as<float>(),color.as<float>(),tiles.as<float>(),W,1080,W,Ht);
  launch_prefix(tiles.as<float>(),prefix_w.as<float>(),host_history?history.as<float>():nullptr,s.a.as<float>(),seed,W,Ht,host_history?1:0);
  blocks[0].window(s.a.as<float>(),s.a.as<float>(),s);
   // Block 0 p1 writes H(v) in f16 (C32 path); crop from the f16 window raster.
   launch_reframe_f32_hin(s.ah.as<__half>(),skip[0].as<float>(),W,Ht,W,Ht,0,0,32,1,0);mark(0,skip[0].as<float>(),N*32);
  float* src=raster[0].as<float>();float* dst=raster[1].as<float>();
  launch_pool_f32(skip[0].as<float>(),src,W,Ht,32,960,576);
  const int first[]={1,5,9,15,23},num[]={4,4,6,8,8};uint w=960,h=576,c=32;
   for(int g=0;g<5;++g){
   const u8* in_f8=nullptr; const __half* in_h=nullptr;
   for(int j=0;j<num[g];++j){int id=first[g]+j;auto&b=blocks[id];bool emit_f8=!b.raw&&b.c>=64;bool emit_h=b.c==32;b.run(src,dst,s,in_f8,emit_f8,in_h,emit_h);std::swap(src,dst);mark(id,src,w*h*c);in_f8=emit_f8?s.out8.as<u8>():nullptr;in_h=emit_h?s.out16.as<__half>():nullptr;}
   hip_check(hipMemcpyAsync(skip[g+1].p,src,size_t(w)*h*c*4,hipMemcpyDeviceToDevice,stream),"encoder skip");
   uint ow=g==4?32:w/2,oh=g==4?20:h/2;
   launch_pool_f32(src,s.b.as<float>(),w,h,c,ow,oh);
   launch_linear_half(s.b.as<float>(),g==4?head_w.as<__half>():ds[g].as<__half>(),dst,ow*oh,c*2,c,1,1,0);
   std::swap(src,dst);w=ow;h=oh;c*=2;
  }
  launch_gather_f32(src,map.as<int>(),dst,655360);std::swap(src,dst);
  for(int j=0;j<8;++j){vit[j].run(src,dst,s);std::swap(src,dst);mark(31+j,src,655360);}
  launch_gather_f32(src,inv.as<int>(),dst,655360);std::swap(src,dst);
  up[0].run(src,skip[5].as<float>(),dst,s);std::swap(src,dst);mark(39,src,60*36*512);
   const int decfirst[]={40,48,56,62,66},decnum[]={8,8,6,4,4};
   for(int g=0;g<5;++g){if(g){up[g].run(src,skip[5-g].as<float>(),dst,s);std::swap(src,dst);}
    const u8* in_f8=nullptr; const __half* in_h=nullptr;
    for(int j=0;j<decnum[g];++j){int id=decfirst[g]+j;auto&b=blocks[id];bool emit_f8=!b.raw&&b.c>=64;bool emit_h=b.c==32;b.run(src,dst,s,in_f8,emit_f8,in_h,emit_h);std::swap(src,dst);mark(id,src,b.w*b.h*b.c);in_f8=emit_f8?s.out8.as<u8>():nullptr;in_h=emit_h?s.out16.as<__half>():nullptr;}}
  launch_post_merge_f32(src,skip[0].as<float>(),post_sc.as<float>(),dst,W,Ht,true);std::swap(src,dst);
  blocks[70].run(src,dst,s);std::swap(src,dst);mark(70,src,N*32);
  launch_head_f32(src,head_rgb.as<float>(),color.as<float>(),out.as<float>(),N);
  hip_check(hipGetLastError(),"network launches");hip_check(hipEventRecord(ev1,stream),"end");
  hip_check(hipMemcpyAsync(output,out.p,size_t(N)*12,hipMemcpyDeviceToHost,stream),"output download");hip_check(hipStreamSynchronize(stream),"network completion");hip_check(hipEventElapsedTime(&gpu_ms,ev0,ev1),"GPU elapsed");
  if(tracing){std::vector<float> stats(71*768);std::vector<uint> bits(71*512);trace_stats.download(stats.data(),stats.size()*4);trace_bits.download(bits.data(),bits.size()*4);
   bool bad=false;for(int b=0;b<71;++b){auto&t=trace[b];t.minimum=INFINITY;t.maximum=-INFINITY;t.checked=true;
    for(int l=0;l<256;++l){t.minimum=std::min(t.minimum,stats[b*768+l*3]);t.maximum=std::max(t.maximum,stats[b*768+l*3+1]);t.sum+=stats[b*768+l*3+2];t.nonfinite+=bits[b*512+l*2];t.hash^=bits[b*512+l*2+1];}
    std::fprintf(stderr,"graph block=%02d count=%u n=%u nonfinite=%u min=%.9g max=%.9g sum=%.12g hash=%08x\n",b,counts[b],t.elements,t.nonfinite,t.minimum,t.maximum,t.sum,t.hash);bad|=t.nonfinite!=0||counts[b]!=1;}
   if(bad)throw std::runtime_error("nonfinite/incomplete graph trace");
  }
 }
};
Network70::Network70():impl(new Impl){}
Network70::~Network70()=default;
void Network70::create(const std::string&dir){auto next=std::make_unique<Impl>();next->create(dir);impl.swap(next);}
void Network70::run(const float* in,float* out,uint seed,const float* host_history){std::lock_guard<std::mutex> lock(impl->mutex);try{impl->run(in,out,seed,host_history);}catch(...){if(impl->stream)(void)hipStreamSynchronize(impl->stream);throw;}}
float Network70::last_gpu_ms()const{return impl->gpu_ms;}
std::array<uint,71> Network70::last_block_counts()const{return impl->counts;}
std::array<Network70::Trace,71> Network70::last_trace()const{return impl->trace;}
} // namespace dlss5
