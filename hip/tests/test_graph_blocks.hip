// Compiles the real graph implementation into this test TU to exercise its
// internal complete block scheduler against independent scalar HLSL functions.
#include "../src/network.hip"
#include <random>
#include <cstring>
using namespace dlss5;
static float hh(float x){return float((_Float16)x);}
static float ff(float x){float a=std::fabs(x);if(a==0)return x;if(a>=448)return std::copysign(448.f,x);float step=a<.015625f?1.f/512:std::exp2(std::floor(std::log2(a))-3);return std::copysign(std::nearbyint(a/step)*step,x);}
static float aa(float x){float g=std::clamp(x,-4.f,4.f);return ff(x*(g*(std::fabs(g)*(-.055908203125f)+.447265625f)+.89453125f));}
static float ee(float x){float y=std::clamp(std::fma(x,.044921875f,1.30078125f),1.03125f,1.5693359375f);_Float16 half=(_Float16)y;uint16_t bits;memcpy(&bits,&half,2);bits=uint16_t((unsigned(bits)<<5)+0x8000);memcpy(&half,&bits,2);return float(half);}
static void save(const std::string&dir,const std::string&name,const std::vector<float>&v){std::ofstream f(dir+"/"+name+".f32",std::ios::binary);f.write((char*)v.data(),v.size()*4);}
static std::vector<float> linear(const std::vector<float>&x,const float*w,uint n,uint ci,uint co,const std::vector<float>*res=nullptr,const float*sc=nullptr,int mode=0,bool matrix=false){
 std::vector<float>y(size_t(n)*co);for(uint t=0;t<n;++t)for(uint j=0;j<co;++j){float sum=res&&mode!=4?(matrix?(*res)[t*co+j]*sc[j]:hh((*res)[t*co+j]*sc[j])):0;
 for(uint k=0;k<ci;++k)sum+=ff(x[t*ci+k])*w[j*ci+k];if(res&&mode==4)sum+=(*res)[t*co+j]*sc[j];
 y[t*co+j]=mode==3?sum:mode==2?aa(sum):mode==5?ff(sum):mode==1?ff(hh(sum)):hh(sum);}return y;
}
static std::vector<float> reference(const std::vector<float>&input,const std::vector<float>&fw,const std::vector<float>&aw,uint c,uint shift,bool raw,const std::vector<float>&fp={}){
 uint w=8,h=8,px=(shift&1)?4:0,py=(shift&2)?4:0,sw=(w+px+7)&~7u,sh=(h+py+7)&~7u,n=sw*sh,mat=c*c;
 std::vector<float>x(size_t(n)*c);for(uint y=0;y<h;++y)for(uint xx=0;xx<w;++xx){uint wx=xx+px,wy=y+py,t=((wy/8)*(sw/8)+wx/8)*64+(wy%8)*8+wx%8;for(uint j=0;j<c;++j)x[t*c+j]=input[(y*w+xx)*c+j];}
 std::vector<float>feature;
 if(c==512){auto mix=linear(x,fw.data(),n,512,512,nullptr,nullptr,1);std::vector<float>branch(n*512);
 for(uint t=0;t<n;++t)for(uint g=0;g<8;++g){float hidden[256];for(uint j=0;j<256;++j){float v=0;for(uint k=0;k<64;++k)v+=mix[t*512+g*64+k]*fw[262144+g*16384+j*64+k];hidden[j]=aa(hh(v));}for(uint j=0;j<64;++j){float v=0;for(uint k=0;k<256;++k)v+=hidden[k]*fw[393216+g*16384+j*256+k];branch[t*512+g*64+j]=ff(hh(v));}}
 feature=linear(branch,fp.data(),n,c,c,&x,fp.data()+mat,1);
 }else{size_t off=c==32?512:0;auto hidden=linear(x,fw.data()+off,n,c,4*c,nullptr,nullptr,2);std::vector<float>branch(n*c);
 for(uint t=0;t<n;++t)for(uint j=0;j<c;++j){float sum=c==32?x[t*c+j]*fw[8704+j]:0;for(uint k=0;k<4*c;++k)sum+=hidden[t*4*c+k]*fw[off+4*mat+j*4*c+k];branch[t*c+j]=c==32?hh(sum):ff(sum);}
 feature=c==32?branch:linear(branch,fw.data()+8*mat,n,c,c,&x,fw.data()+9*mat,5);}
 // Configured multihead QKV uses FP32 squares; only C32 uses half squares.
 auto q=linear(feature,aw.data(),n,c,3*c,nullptr,nullptr,3);
 for(uint t=0;t<n;++t)for(uint part=0;part<3;++part)for(uint head=0;head<c/32;++head){uint base=t*3*c+part*c+head*32;float sum=0;for(uint j=0;j<32;++j)sum+=c==32?hh(q[base+j]*q[base+j]):q[base+j]*q[base+j];float gain=part==2?1:1/std::sqrt(std::max(sum,6.198883056640625e-5f))*(part==0?aw[4*mat+c/32*4096+head]:1);for(uint j=0;j<32;++j)q[base+j]=ff(q[base+j]*gain);}
 std::vector<float>att(n*c);for(uint win=0;win<n/64;++win)for(uint head=0;head<c/32;++head)for(uint t=0;t<64;++t){float p[64],den=0;for(uint k=0;k<64;++k){float sum=0;for(uint j=0;j<32;++j)sum+=q[(win*64+t)*3*c+head*32+j]*q[(win*64+k)*3*c+c+head*32+j];p[k]=ee(sum+aw[4*mat+head*4096+t*64+k]);den+=p[k];}for(uint j=0;j<32;++j){float sum=0;for(uint k=0;k<64;++k)sum+=ff(p[k]/den)*q[(win*64+k)*3*c+2*c+head*32+j];att[(win*64+t)*c+head*32+j]=ff(sum);}}
 auto result=linear(att,aw.data()+3*mat,n,c,c,&feature,aw.data()+4*mat+c/32*4096+c/32,c==32?4:raw?0:c==512?1:5,c!=32&&c!=512);
 std::vector<float>r(64*c);for(uint y=0;y<h;++y)for(uint xx=0;xx<w;++xx){uint wx=xx+px,wy=y+py,t=((wy/8)*(sw/8)+wx/8)*64+(wy%8)*8+wx%8;for(uint j=0;j<c;++j)r[(y*w+xx)*c+j]=result[t*c+j];}return r;
}
int main(){try{const std::string dir=std::string(std::getenv("DLSS5_GRAPH_TEST_BUILD")?std::getenv("DLSS5_GRAPH_TEST_BUILD"):"linux/build/graph-audit")+"/block-fixture";std::filesystem::create_directories(dir);Scratch scratch;scratch.create();unsigned bad=0;
 for(uint c:{32u,64u,128u,256u,512u})for(uint shift:{0u,1u,2u,3u})for(bool raw:{false,true}){uint mat=c*c;std::mt19937 rng(500+c);std::vector<float>fw(c==512?524288:c==32?8736:9*mat+c),aw(4*mat+(c/32)*4096+c/32+c),x(64*c),fp(c==512?262656:0);
 for(auto&v:fw)v=ff(float(int(rng()%15)-7)/32);for(auto&v:aw)v=ff(float(int(rng()%15)-7)/32);for(auto&v:x)v=hh(float(int(rng()%21)-10)*.111f);
 for(auto&v:fp)v=ff(float(int(rng()%15)-7)/32);
 for(uint j=0;j<c;++j){if(c!=512)fw[fw.size()-c+j]=hh(.71f+j*.003f);else fp[mat+j]=hh(.71f+j*.003f);aw[aw.size()-c+j]=hh(.57f+j*.002f);}for(uint j=0;j<c/32;++j)aw[4*mat+c/32*4096+j]=hh(1.5f+j*.31f);
 int id=c==32?0:c==64?5:c==128?9:c==256?15:23;auto name="block"+std::to_string(id);save(dir,name+(c==512?"-ffwd":"-ffn"),fw);save(dir,name+"-attention",aw);if(c==512)save(dir,name+"-ffwd-projection",fp);
 Block b;b.create(dir,id,c,8,8,shift,raw);DeviceBuf di(x.size()*4),dout(x.size()*4);di.upload(x.data(),x.size()*4);b.run(di.as<float>(),dout.as<float>(),scratch);hip_check(hipDeviceSynchronize(),"block");std::vector<float>got(x.size());dout.download(got.data(),got.size()*4);auto want=reference(x,fw,aw,c,shift,raw,fp);unsigned mismatches=0;float maxerr=0;
 for(size_t i=0;i<x.size();++i)if(!std::isfinite(got[i])||got[i]!=want[i]){if(mismatches<2)std::printf("C%u shift%u i%zu got=%g expected=%g\n",c,shift,i,got[i],want[i]);++mismatches;maxerr=std::max(maxerr,std::fabs(got[i]-want[i]));}bad+=mismatches;
 std::printf("complete C%u shift%u raw%d mismatches=%u max_error=%g\n",c,shift,int(raw),mismatches,maxerr);
 }return bad?1:0;}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
