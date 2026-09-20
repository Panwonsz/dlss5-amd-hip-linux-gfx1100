// Independent scalar fast-HLSL oracles: no production H/F/activation calls.
#include "dlss5_launch.hpp"
#include "dlss5_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
using namespace dlss5;
static float h(float x){return float((_Float16)x);}
static float f(float x){float a=std::fabs(x);if(a>=448)return std::copysign(448.f,x);float step=a<.015625f?1.f/512:std::exp2(std::floor(std::log2(a))-3);return a==0?x:std::copysign(std::nearbyint(a/step)*step,x);}
static float act(float x){float g=std::clamp(x,-4.f,4.f);return f(x*(g*(std::fabs(g)*(-.055908203125f)+.447265625f)+.89453125f));}
static int bad=0;static void same(float a,float b,const char* label){if(!std::isfinite(a)||a!=b){if(bad<8)std::printf("%s got=%g want=%g\n",label,a,b);++bad;}}
static DeviceBuf upload(const std::vector<float>&v){DeviceBuf d(v.size()*4);d.upload(v.data(),v.size()*4);return d;}
static std::vector<float> download(DeviceBuf&d,size_t n){std::vector<float> v(n);d.download(v.data(),n*4);return v;}
static void partitions(){
 const uint m=16,n=32,k=4096;std::vector<float> a(m*k),b(n*k),r(m*n),sc(n,.731f);
 std::mt19937 rng(51);for(auto&v:a)v=f(float(int(rng()%31)-15)/32);for(auto&v:b)v=f(float(int(rng()%31)-15)/128);for(auto&v:r)v=h(float(int(rng()%31)-15)/16);
 auto da=upload(a),db=upload_tiled_weights(b.data(),n,k),dr=upload(r),ds=upload(sc);DeviceBuf out(m*n*4);
 // ordered=2 means independent four-partition FP32 reduction, not K32 half rounding.
 launch_linear_f32(da.as<float>(),db.as<u8>(),out.as<float>(),m,n,k,dr.as<float>(),ds.as<float>(),1,2);hip_check(hipDeviceSynchronize(),"partitions");auto y=download(out,m*n);
 for(uint t=0;t<m;++t)for(uint j=0;j<n;++j){float total=h(r[t*n+j]*sc[j]);for(uint p=0;p<4;++p){float part=0;for(uint q=p*(k/4);q<(p+1)*(k/4);++q)part+=a[t*k+q]*b[j*k+q];total+=part;}same(y[t*n+j],f(h(total)),"ViT partition");}
}
static void split(){const uint n=16;std::vector<float>a(n*512),pre(512*512),ex(8*16384),ct(ex.size());std::mt19937 rng(713);
 for(auto&v:a)v=f(float(int(rng()%31)-15)/16);for(auto&v:pre)v=f(float(int(rng()%7)-3)/64);for(auto&v:ex)v=f(float(int(rng()%31)-15)/64);for(auto&v:ct)v=f(float(int(rng()%7)-3)/32);
 auto da=upload(a),dp=upload_tiled_e4m3(pre.data(),512,512);std::vector<u8>et(ex.size()),tt(ct.size());for(int g=0;g<8;++g){pack_tiled_e4m3(et.data()+g*16384,ex.data()+g*16384,256,64);pack_tiled_e4m3(tt.data()+g*16384,ct.data()+g*16384,64,256);}DeviceBuf de(et.size()),dc(tt.size()),dy(n*512*4);de.upload(et.data(),et.size());dc.upload(tt.data(),tt.size());
 launch_split_f32(da.as<float>(),dp.as<u8>(),de.as<u8>(),dc.as<u8>(),dy.as<float>(),n);hip_check(hipDeviceSynchronize(),"split");auto y=download(dy,n*512);unsigned distinguish=0;
 for(uint t=0;t<n;++t){std::vector<float> mix(512),hidden(256);for(uint j=0;j<512;++j){float v=0;for(uint k=0;k<512;++k)v+=a[t*512+k]*pre[j*512+k];mix[j]=f(h(v));}
 for(uint g=0;g<8;++g){for(uint j=0;j<256;++j){float v=0;for(uint k=0;k<64;++k)v+=mix[g*64+k]*ex[g*16384+j*64+k];hidden[j]=act(h(v));distinguish+=hidden[j]!=act(v);}
 for(uint j=0;j<64;++j){float v=0;for(uint k=0;k<256;++k)v+=hidden[k]*ct[g*16384+j*256+k];same(y[t*512+g*64+j],f(h(v)),"split logical");}}}
 if(!distinguish){std::puts("split fixture does not distinguish hidden H");++bad;}std::printf("split hidden H distinguishing cases=%u\n",distinguish);
}
static void half_linear(){
 // Dyadic non-FP8 half values make scalar and WMMA reductions exactly
 // comparable; arbitrary half products can differ by one final half ULP
 // because hardware and scalar float32 summation trees are not identical.
 for(uint k:{32u,64u,128u,256u,512u,1024u}){const uint m=19,n=32;std::vector<float>a(m*k),b(n*k);std::vector<__half>tile(n*k);for(size_t i=0;i<a.size();++i)a[i]=h(float(int(i%31)-15)*.01025390625f);for(size_t i=0;i<b.size();++i)b[i]=h(float(int((i*7+i/19)%29)-14)*.004150390625f);
 for(uint j=0;j<n;++j)for(uint q=0;q<k;++q)tile[((j/16)*(k/32)+q/32)*512+(q%32)*16+j%16]=__float2half(b[j*k+q]);auto da=upload(a);DeviceBuf db(tile.size()*2),dy(m*n*4);db.upload(tile.data(),tile.size()*2);
 for(int ordered:{0,1})for(int raw:{0,1,2}){uint parts=k==1024?4:1;launch_linear_half(da.as<float>(),db.as<__half>(),dy.as<float>(),m,n,k,parts,ordered,raw);hip_check(hipDeviceSynchronize(),"half matmul");auto y=download(dy,m*n);
 for(uint t=0;t<m;++t)for(uint j=0;j<n;++j){float total=0;for(uint p=0;p<parts;++p){float v=0;for(uint g=p*(k/parts);g<(p+1)*(k/parts);g+=32){if(ordered){float z=0;for(uint q=g;q<g+32;++q)z+=a[t*k+q]*b[j*k+q];v=h(v+z);}else for(uint q=g;q<g+32;++q)v+=a[t*k+q]*b[j*k+q];}total+=v;}float e=raw==2?total:raw==1?h(total):f(h(total));if(raw==2){if(!std::isfinite(y[t*n+j])||std::fabs(y[t*n+j]-e)>2e-6f){++bad;}}else same(y[t*n+j],e,"half matmul");}}
 }
}
static void trace_hash(){
 std::vector<float>a(1024);for(size_t i=0;i<a.size();++i)a[i]=float(i)/128;
 auto da=upload(a);DeviceBuf stats(768*4),bits(512*4);std::vector<uint>b(512);
 auto digest=[&](){launch_trace_f32(da.as<float>(),a.size(),stats.as<float>(),bits.as<uint>());hip_check(hipDeviceSynchronize(),"trace");bits.download(b.data(),b.size()*4);uint hash=0;for(uint i=0;i<256;++i){if(b[i*2])++bad;hash^=b[i*2+1];}return hash;};
 uint first=digest();std::swap(a[0],a[17]);da.upload(a.data(),a.size()*4);uint second=digest();
 if(first==second){std::puts("trace permutation hash collision");++bad;}
}
static void seams(){
 const uint iw=32,ih=20,w=60,ht=36,c=32;std::vector<float>lo(iw*ih*c),skip(w*ht*c),sc(c),weights(96),color(w*ht*4),ps(64);
 for(size_t i=0;i<lo.size();++i)lo[i]=h(float(int(i%29)-14)*.00371f);for(size_t i=0;i<skip.size();++i)skip[i]=h(float(int(i%31)-15)*.0371f);for(uint j=0;j<c;++j){sc[j]=h(.711f+j*.013f);ps[j]=h(.7f+j*.011f);ps[32+j]=h(.31f+j*.007f);}
 for(size_t i=0;i<weights.size();++i)weights[i]=h(float(int(i%11)-5)*.173f);for(size_t i=0;i<color.size();++i)color[i]=float(i%41)/40;
 auto dl=upload(lo),ds=upload(skip),dsc=upload(sc),dw=upload(weights),dr=upload(color),dps=upload(ps);DeviceBuf dy(skip.size()*4),rgb(w*ht*3*4);
 launch_up_f32(dl.as<float>(),ds.as<float>(),dsc.as<float>(),dy.as<float>(),iw,ih,w,ht,c);hip_check(hipDeviceSynchronize(),"up");auto y=download(dy,skip.size());
 for(uint p=0;p<w*ht;++p)for(uint j=0;j<c;++j)same(y[p*c+j],h(lo[((p/w/2)*iw+(p%w)/2)*c+j]+f(skip[p*c+j])*sc[j]),"raw up/crop");
 launch_head_f32(dy.as<float>(),dw.as<float>(),dr.as<float>(),rgb.as<float>(),w*ht);hip_check(hipDeviceSynchronize(),"head");auto z=download(rgb,w*ht*3);
 for(uint p=0;p<w*ht;++p)for(uint row=0;row<3;++row){float a=0;for(uint j=0;j<c;++j)a+=y[p*c+j]*weights[row*c+j];float base=color[p*4+row]*.125f-.0625f;float e=std::clamp((h(a)*.03125f+base)*8.f+.5f,0.f,1.f);same(z[p*3+row],e,"RGB head");}
 // Post70 has two distinct learned scales and no FP8 output at merge.
 launch_post_merge_f32(dl.as<float>(),ds.as<float>(),dps.as<float>(),dy.as<float>(),w,ht);hip_check(hipDeviceSynchronize(),"post merge");y=download(dy,skip.size());
 for(uint p=0;p<w*ht;++p)for(uint j=0;j<c;++j)same(y[p*c+j],h(h(lo[((p/w/2)*(w/2)+(p%w)/2)*c+j]*ps[j])+f(skip[p*c+j])*ps[32+j]),"post merge");
 // Both reflected axes, nontrivial padding, and the prefix's true tile layout.
 std::vector<float>input(13*9*4);for(size_t i=0;i<input.size();++i)input[i]=float(i);auto di=upload(input);DeviceBuf raster(24*16*16),tiles(24*16*16);
 launch_rgb_graph(di.as<float>(),raster.as<float>(),tiles.as<float>(),13,9,24,16);hip_check(hipDeviceSynchronize(),"reflect");auto rr=download(raster,24*16*4),tt=download(tiles,24*16*4);
 for(uint yy=0;yy<16;++yy)for(uint xx=0;xx<24;++xx){uint rx=xx%24,ry=yy%16;rx=rx<13?rx:24-rx;ry=ry<9?ry:16-ry;uint tile=((yy/8)*3+xx/8)*64+(yy%8)*8+xx%8;for(uint j=0;j<4;++j){float e=input[(ry*13+rx)*4+j];same(rr[(yy*24+xx)*4+j],e,"reflect raster");same(tt[tile*4+j],e,"reflect tile");}}
}
int main(int argc,char**argv){if(argc>1&&std::string(argv[1])=="partition")partitions();else{partitions();split();half_linear();seams();trace_hash();}std::printf("graph operators mismatches=%d\n",bad);return bad?1:0;}
