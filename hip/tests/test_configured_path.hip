// Configured game HLSL seams. Synthetic operator inputs, NOT actual model weights.
#include "../src/network.hip"
#include <random>
#include <cstring>
using namespace dlss5;
static float hh(float x){return float((_Float16)x);}
static float ff(float x){float a=std::fabs(x);if(a==0)return x;if(a>=448)return std::copysign(448.f,x);float step=a<.015625f?1.f/512:std::exp2(std::floor(std::log2(a))-3);return std::copysign(std::nearbyint(a/step)*step,x);}
static float ee(float x){_Float16 a=(_Float16)std::clamp(std::fma(x,.044921875f,1.30078125f),1.03125f,1.5693359375f);uint16_t b;memcpy(&b,&a,2);b=uint16_t((unsigned(b)<<5)+0x8000);memcpy(&a,&b,2);return float(a);}
static unsigned failures=0;
static void check(const char* label,size_t bad){printf("%s mismatches=%zu\n",label,bad);failures+=bad!=0;}
static void save(const std::string&d,const std::string&n,const std::vector<float>&v){std::ofstream f(d+"/"+n+".f32",std::ios::binary);f.write((char*)v.data(),v.size()*4);}
static void linear(const float*a,const u8*b,float*y,uint c,const float*r=nullptr,const float*s=nullptr,bool direct=true,bool matrix=false){
#ifdef DLSS5_CONFIGURED_POLICIES
 launch_linear_f32(a,b,y,16,c,c,r,s,direct?5:1,0,matrix);
#else
 launch_linear_f32(a,b,y,16,c,c,r,s,1,0);
#endif
}
static void norm(const float*q,const float*s,u8*y,uint c,bool half){
#ifdef DLSS5_CONFIGURED_POLICIES
 launch_normalize_qkv(q,s,y,16,c,1.f,half);
#else
 launch_normalize_qkv(q,s,y,16,c,1.f);
#endif
}
static void attention(uint c,const u8*q,const float*b,u8*y,bool direct){
#ifdef DLSS5_CONFIGURED_POLICIES
 launch_attention(c,q,b,y,512,8,1,direct);
#else
 launch_attention(c,q,b,y,512,8,1);
#endif
}
__global__ void poly_probe(unsigned* out){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=2097152)return;float x=(float(i)-1048576.f)/262144.f;volatile float q=fabsf(x)*(-.055908203125f);volatile float qq=q+.447265625f;volatile float p=x*qq;volatile float pp=p+.89453125f;float exact=x*pp;
#ifdef DLSS5_CONFIGURED_POLICIES
 u8 got=e4m3_byte(ActivatePolyC32(x));
#else
 u8 got=e4m3_byte(ActivatePoly(x));
#endif
 out[i]=unsigned(got)|(unsigned(e4m3_byte(exact))<<8);}
int main(){try{
 const char* env=std::getenv("DLSS5_GRAPH_TEST_BUILD");std::string dir=std::string(env?env:"linux/build/fix-review")+"/configured-synthetic";std::filesystem::create_directories(dir);
 Scratch scratch;scratch.create();
 // F1 checks the REAL configured Network70 post stage, not an arbitrary shift.
 {Network70 net;net.create("linux/build/weights-complete/tables");auto&b=net.impl->blocks[70];check("F1 graph post shift3",b.px!=4||b.py!=4);
 const uint w=16,c=32,px=b.px,py=b.py,sw=(w+px+7)&~7u,sh=(w+py+7)&~7u;std::vector<float>x(w*w*c),y(x.size());for(uint yy=0;yy<w;++yy)x[(yy*w+7)*c]=1;
 DeviceBuf dx(x.size()*4),dy(y.size()*4),q(sw*sh*96),att(sw*sh*32),bias(4096*4);dx.upload(x.data(),x.size()*4);bias.zero();launch_reframe_f32(dx.as<float>(),scratch.a.as<float>(),w,w,sw,sh,px,py,c,0,0);std::vector<float>fr(sw*sh*c);scratch.a.download(fr.data(),fr.size()*4);std::vector<u8>qh(sw*sh*96);for(uint t=0;t<sw*sh;++t)for(uint j=0;j<c;++j)qh[t*96+64+j]=e4m3_byte(fr[t*c+j]);q.upload(qh.data(),qh.size());launch_attention(32,q.as<u8>(),bias.as<float>(),att.as<u8>(),sw,sh,1);launch_f8_to_f32(att.as<u8>(),scratch.d.as<float>(),sw*sh*c);launch_reframe_f32(scratch.d.as<float>(),dy.as<float>(),w,w,sw,sh,px,py,c,1,0);dy.download(y.data(),y.size()*4);check("F1 boundary impulse",y[(4*w+8)*c]!=.125f);}
 // F2: true Block scheduling; first/pre/post remain raw, only six raw-chain IDs use F(input).
 {std::vector<float>fw(8736),aw(8225);std::fill(fw.end()-32,fw.end(),.75f);std::fill(aw.end()-32,aw.end(),1.f);std::vector<float>x(64*32,1.03125f),y(x.size());DeviceBuf dx(x.size()*4),dy(x.size()*4);dx.upload(x.data(),x.size()*4);
 for(int id:{0,1,2,3,4,66,67,68,69,70}){std::string p=id==70?"post70":"block"+std::to_string(id);save(dir,p+"-ffn",fw);save(dir,p+"-attention",aw);Block b;b.create(dir,id,32,8,8,0,true);b.run(dx.as<float>(),dy.as<float>(),scratch);dy.download(y.data(),y.size()*4);bool chain=id==2||id==3||id==4||id==67||id==68||id==69;size_t bad=0;for(float v:y)bad+=v!=hh((chain?ff(x[0]):x[0])*.75f);check(("F2 block"+std::to_string(id)).c_str(),bad);}
 Block b;b.create(dir,2,32,8,8,0,true);b.run(dx.as<float>(),dy.as<float>(),scratch);b.create(dir,3,32,8,8,0,true);b.run(dy.as<float>(),dx.as<float>(),scratch);dx.download(y.data(),y.size()*4);size_t bad=0;for(float v:y)bad+=v!=hh(ff(hh(ff(x[0])*.75f))*.75f);check("F2 consecutive chain",bad);}
 {std::vector<float>lo(32,1.03125f),pre(128,.3125f),sc(64,.5f),y(128);std::fill(sc.begin(),sc.begin()+32,.75f);DeviceBuf dl(128),dp(512),ds(256),dy(512);dl.upload(lo.data(),128);dp.upload(pre.data(),512);ds.upload(sc.data(),256);
#ifdef DLSS5_CONFIGURED_POLICIES
 launch_post_merge_f32(dl.as<float>(),dp.as<float>(),ds.as<float>(),dy.as<float>(),2,2,true);
#else
 launch_post_merge_f32(dl.as<float>(),dp.as<float>(),ds.as<float>(),dy.as<float>(),2,2);
#endif
 dy.download(y.data(),512);size_t bad=0;for(float v:y)bad+=v!=hh(hh(ff(lo[0])*.75f)+ff(pre[0])*.5f);check("F3 main8 low before scale",bad);}
 for(uint c:{64u,128u,256u,512u}){std::vector<float>a(16*c),b(c*c),y(16*c),r(16*c,1.125f),s(c,.94482421875f);for(uint t=0;t<16;++t){a[t*c]=1;a[t*c+1]=.0625f;a[t*c+2]=.001953125f;}for(uint j=0;j<c;++j){b[j*c]=1;b[j*c+1]=1;b[j*c+2]=.125f;}DeviceBuf da(a.size()*4),dy(y.size()*4),dr(r.size()*4),ds(s.size()*4);da.upload(a.data(),a.size()*4);dr.upload(r.data(),r.size()*4);ds.upload(s.data(),s.size()*4);auto db=upload_tiled_weights(b.data(),c,c);
 for(bool direct:{false,true}){linear(da.as<float>(),db.as<u8>(),dy.as<float>(),c,nullptr,nullptr,direct);dy.download(y.data(),y.size()*4);size_t bad=0;for(float v:y)bad+=v!=(direct?1.125f:1.f);check(("F4 projection C"+std::to_string(c)+(direct?" direct":" half")).c_str(),bad);}
 std::fill(b.begin(),b.end(),0);db=upload_tiled_weights(b.data(),c,c);for(bool matrix:{false,true}){linear(da.as<float>(),db.as<u8>(),dy.as<float>(),c,dr.as<float>(),ds.as<float>(),true,matrix);dy.download(y.data(),y.size()*4);size_t bad=0;float v=r[0]*s[0],want=ff(matrix?v:hh(v));for(float z:y)bad+=z!=want;check(("F4 residual C"+std::to_string(c)+(matrix?" matrix":" half")).c_str(),bad);}}
 // Independent attention oracle exercises midpoint-adjacent AV accumulators.
 for(uint c:{32u,64u,128u,256u,512u}){const uint nw=64;std::vector<u8>q(nw*64*3*c),out(nw*64*c);std::vector<float>bias((c/32)*4096),p((c/32)*4096);std::mt19937 rng(817);for(float&v:bias)v=hh(float(int(rng()%129)-64)/16.f);for(uint t=0;t<nw*64;++t)for(uint j=0;j<c;++j)q[t*3*c+2*c+j]=e4m3_byte(float(int(rng()%63)-31)/16.f);
 for(uint head=0;head<c/32;++head)for(uint t=0;t<64;++t){float sum=0;for(uint k=0;k<64;++k){p[head*4096+t*64+k]=ee(bias[head*4096+t*64+k]);sum+=p[head*4096+t*64+k];}for(uint k=0;k<64;++k)p[head*4096+t*64+k]=ff(p[head*4096+t*64+k]/sum);}
 DeviceBuf dq(q.size()),db(bias.size()*4),dy(out.size());dq.upload(q.data(),q.size());db.upload(bias.data(),bias.size()*4);for(bool direct:{false,true}){attention(c,dq.as<u8>(),db.as<float>(),dy.as<u8>(),direct);dy.download(out.data(),out.size());size_t bad=0;for(uint w=0;w<nw;++w)for(uint t=0;t<64;++t)for(uint j=0;j<c;++j){float sum=0;for(uint k=0;k<64;++k)sum+=p[(j/32)*4096+t*64+k]*from_e4m3(q[(w*64+k)*3*c+2*c+j]);bad+=from_e4m3(out[(w*64+t)*c+j])!=ff(direct?sum:hh(sum));}check(("F4 AV C"+std::to_string(c)+(direct?" direct":" half")).c_str(),bad);}}
 for(uint c:{32u,64u,128u,256u,512u,1024u})for(float value:{.333251953125f,300.f}){std::vector<float>q(16*3*c,value),sc(c/32,1);std::vector<u8>y(q.size());DeviceBuf dq(q.size()*4),ds(sc.size()*4),dy(y.size());dq.upload(q.data(),q.size()*4);ds.upload(sc.data(),sc.size()*4);for(bool half:{false,true}){norm(dq.as<float>(),ds.as<float>(),dy.as<u8>(),c,half);dy.download(y.data(),y.size());float sum=0;for(uint j=0;j<32;++j)sum+=half?hh(value*value):value*value;float expected=ff(value/std::sqrt(std::max(sum,6.198883056640625e-5f)));size_t bad=0;for(uint t=0;t<16;++t)for(uint j=0;j<2*c;++j)bad+=from_e4m3(y[t*3*c+j])!=expected;check(("F5 norm C"+std::to_string(c)+" value"+std::to_string(value)+(half?" half":" float")).c_str(),bad);}}
 {const uint n=2097152;DeviceBuf d(n*4);poly_probe<<<n/256,256>>>(d.as<unsigned>());std::vector<unsigned>v(n);d.download(v.data(),n*4);size_t bad=0;for(unsigned z:v)bad+=(z&255)!=(z>>8);check("C32 precise ActivatePoly bytes",bad);}
 hip_check(hipDeviceSynchronize(),"configured seams");printf("configured seam failing groups=%u\n",failures);return failures?1:0;
 }catch(const std::exception&e){fprintf(stderr,"configured test error: %s\n",e.what());return 2;}}
