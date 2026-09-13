// Full 640-token x 1024-channel ViT block, sparse nonzero logical tables.
// CPU oracle independently computes fast HLSL math; no production H/F helpers.
#include "../src/network.hip"
#include <cstring>
using namespace dlss5;
static float h(float x){return float((_Float16)x);}
static float f(float x){float a=std::fabs(x);if(a==0)return x;if(a>=448)return std::copysign(448.f,x);float step=a<.015625f?1.f/512:std::exp2(std::floor(std::log2(a))-3);return std::copysign(std::nearbyint(a/step)*step,x);}
static float activation(float x){float g=std::clamp(x,-4.f,4.f);return f(x*(g*(std::fabs(g)*(-.055908203125f)+.447265625f)+.89453125f));}
static float expmap(float x){_Float16 slope;uint16_t bits=0x2dbb;std::memcpy(&slope,&bits,2);_Float16 a=(_Float16)std::clamp(std::fma(x,float(slope),1.708984375f),1.439453125f,1.9775390625f);std::memcpy(&bits,&a,2);bits=uint16_t((unsigned(bits)<<4)+0x4000);std::memcpy(&a,&bits,2);return float(a);}
static void save(const std::string&d,const char*n,const std::vector<float>&v){std::ofstream s(d+"/block31-"+n+".f32",std::ios::binary);s.write((char*)v.data(),v.size()*4);}
int main(){try{
 const std::string dir=std::string(std::getenv("DLSS5_GRAPH_TEST_BUILD")?std::getenv("DLSS5_GRAPH_TEST_BUILD"):"linux/build/graph-audit")+"/vit-fixture";std::filesystem::create_directories(dir);
 std::vector<float>ew(4194304),cw(4195328),qw(3145760),pw(1049600),x(655360),expected(x.size());
 for(uint j=0;j<4096;++j)ew[j*1024+j%1024]=.25f*(1+j/1024);
 for(uint j=0;j<1024;++j){for(uint p=0;p<4;++p)cw[j*4096+p*1024+j]=.0625f*(p+1);cw[4194304+j]=h(.5f+(j%7)*.031f);pw[j*1024+(j+7)%1024]=.125f;pw[1048576+j]=h(.31f+(j%11)*.017f);for(uint part=0;part<3;++part)qw[(part*1024+j)*1024+(j+part*3)%1024]=h(.312744140625f+part*.03125f);}
 for(uint head=0;head<32;++head)qw[3145728+head]=h(.37f+head*.023f);
 for(uint t=0;t<640;++t)for(uint j=0;j<1024;++j)x[t*1024+j]=f(float(int((t*7+j*3)%23)-11)*.03125f);
 save(dir,"expand",ew);save(dir,"contract",cw);save(dir,"qkv",qw);save(dir,"projection",pw);
 Vit vit;vit.create(dir,31);Scratch s;s.create();DeviceBuf dx(x.size()*4),dy(x.size()*4);dx.upload(x.data(),x.size()*4);vit.run(dx.as<float>(),dy.as<float>(),s);hip_check(hipDeviceSynchronize(),"vit block");std::vector<float>got(x.size());dy.download(got.data(),got.size()*4);
 std::vector<float>feature(x.size()),q(640*3072),att(x.size());
 for(uint t=0;t<640;++t)for(uint j=0;j<1024;++j){float a=h(x[t*1024+j]*cw[4194304+j]);for(uint p=0;p<4;++p)a+=activation(x[t*1024+j]*.25f*(p+1))*(.0625f*(p+1));feature[t*1024+j]=f(h(a));}
 for(uint t=0;t<640;++t)for(uint part=0;part<3;++part)for(uint head=0;head<32;++head){float values[32],sum=0;for(uint j=0;j<32;++j){uint col=head*32+j;values[j]=feature[t*1024+(col+part*3)%1024]*h(.312744140625f+part*.03125f);sum+=h(values[j]*values[j]);}float gain=part==2?1:1/std::sqrt(std::max(sum,6.198883056640625e-5f))*(part==0?qw[3145728+head]*5.65625f:1);for(uint j=0;j<32;++j)q[t*3072+part*1024+head*32+j]=f(values[j]*gain);}
 for(uint head=0;head<32;++head)for(uint t=0;t<640;++t){float ex[640],den=0;for(uint k=0;k<640;++k){float dot=0;for(uint j=0;j<32;++j)dot+=q[t*3072+head*32+j]*q[k*3072+1024+head*32+j];ex[k]=expmap(dot);den+=ex[k];ex[k]=f(ex[k]);}for(uint j=0;j<32;++j){float a=0;for(uint k=0;k<640;++k)a+=ex[k]*q[k*3072+2048+head*32+j];att[t*1024+head*32+j]=f(h(a/den));}}
 unsigned bad=0;float maxerr=0;for(uint t=0;t<640;++t)for(uint j=0;j<1024;++j){float a=h(feature[t*1024+j]*pw[1048576+j])+att[t*1024+(j+7)%1024]*.125f;expected[t*1024+j]=f(h(a));float error=std::fabs(got[t*1024+j]-expected[t*1024+j]);if(!std::isfinite(got[t*1024+j])||error){if(bad<4)std::printf("vit t=%u c=%u got=%g expected=%g\n",t,j,got[t*1024+j],expected[t*1024+j]);++bad;maxerr=std::max(maxerr,error);}}
 std::printf("complete ViT640x1024 nonzero sparse CPU oracle mismatches=%u max_error=%g\n",bad,maxerr);return bad?1:0;
 }catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
