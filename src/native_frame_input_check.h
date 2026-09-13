#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <limits>
// Portable pixel helpers: no Windows/D3D dependency.
inline uint16_t NativeFloatToHalf(float v){
 uint32_t b;std::memcpy(&b,&v,sizeof b);
 const uint32_t sign=(b>>16)&0x8000u,exp=(b>>23)&255u,mant=b&0x7fffffu;
 if(exp==255)return uint16_t(sign|0x7c00u|(mant?((mant>>13)|0x200u):0u));
 if(exp>142)return uint16_t(sign|0x7c00u);
 if(exp<102)return uint16_t(sign);
 // Round to nearest, ties to even; retain discarded subnormal bits.
 const unsigned shift=exp<113 ? 126-exp : 13;
 const uint32_t value=exp<113 ? mant|0x800000u : mant;
 uint32_t rounded=value>>shift;
 const uint32_t remainder=value&((1u<<shift)-1),halfway=1u<<(shift-1);
 if(remainder>halfway||(remainder==halfway&&(rounded&1)))++rounded;
 // Addition, not OR: a mantissa carry must increment the exponent.
 return uint16_t(sign|((exp<113 ? 0u : (exp-112)<<10)+rounded));
}

enum class NativePixelFormat { unsupported, rgba8_unorm, bgra8_unorm, rgba16_unorm, rgba16_float };
// DXGI values, without Windows headers. Pass the actual typed view format.
// Typeless and sRGB require an explicit view/color-space contract; reject them.
inline NativePixelFormat NativePixelFormatFromDxgi(unsigned format){
 switch(format){
 case 28:return NativePixelFormat::rgba8_unorm;
 case 87:return NativePixelFormat::bgra8_unorm;
 case 11:return NativePixelFormat::rgba16_unorm;
 case 10:return NativePixelFormat::rgba16_float;
 default:return NativePixelFormat::unsupported;
 }
}
inline size_t NativePixelBytes(NativePixelFormat f){
 switch(f){
 case NativePixelFormat::rgba8_unorm:case NativePixelFormat::bgra8_unorm:return 4;
 case NativePixelFormat::rgba16_unorm:case NativePixelFormat::rgba16_float:return 8;
 default:return 0;
 }
}
inline bool NativePixelGeometry(unsigned w,unsigned h){return w&&h&&w<=7680&&h<=4320;}
inline float NativePixelHalfToFloat(uint16_t h){
 const unsigned e=(h>>10)&31,m=h&1023;
 float v=e==0?std::ldexp(float(m),-24):e==31?
  (m?std::numeric_limits<float>::quiet_NaN():std::numeric_limits<float>::infinity()):
  std::ldexp(float(1024+m),int(e)-25);
 return h&0x8000?-v:v;
}
inline float NativePixelChannel(const unsigned char*p,NativePixelFormat f,unsigned c){
 if(f==NativePixelFormat::bgra8_unorm&&c!=3)c=2-c;
 if(NativePixelBytes(f)==4)return p[c]/255.f;
 // D3D packed pixels are little-endian, regardless of host alignment.
 const uint16_t h=uint16_t(p[c*2])|uint16_t(p[c*2+1])<<8;
 return f==NativePixelFormat::rgba16_float?NativePixelHalfToFloat(h):h/65535.f;
}
// Packed rows. Caller supplies width*height*NativePixelBytes(format) readable bytes.
// Failure leaves rgba untouched; no magnitude-based format inference.
inline bool NativePixelsToRgbaF32(const unsigned char*pix,NativePixelFormat format,unsigned width,unsigned height,std::vector<float>&rgba){
 const size_t bpp=NativePixelBytes(format);
 if(!pix||!bpp||!NativePixelGeometry(width,height))return false;
 const size_t n=size_t(width)*height;
 if(format==NativePixelFormat::rgba16_float)
  for(size_t i=0;i<n;i++)for(unsigned c=0;c<4;c++)
   if(!std::isfinite(NativePixelChannel(pix+i*bpp,format,c)))return false;
 rgba.resize(n*4);
 for(size_t i=0;i<n;i++)for(unsigned c=0;c<4;c++)rgba[i*4+c]=NativePixelChannel(pix+i*bpp,format,c);
 return true;
}
// In-place RGB replacement: destination MUST contain the original packed pixels.
// Alpha bytes remain bit-exact. Reject nonfinite/unrepresentable FP16 output
// before writing any pixel. UNORM clamps only finite values; no NaN sanitizing.
inline bool NativeRgbF32ToPixels(const float*rgb,unsigned char*pix,NativePixelFormat format,unsigned width,unsigned height){
 const size_t bpp=NativePixelBytes(format);
 if(!rgb||!pix||!bpp||!NativePixelGeometry(width,height))return false;
 const size_t n=size_t(width)*height;
 for(size_t i=0;i<n*3;i++){
  if(!std::isfinite(rgb[i]))return false;
  if(format==NativePixelFormat::rgba16_float&&(NativeFloatToHalf(rgb[i])&0x7c00)==0x7c00)return false;
 }
 if(format==NativePixelFormat::rgba16_float)
  for(size_t i=0;i<n;i++)if(!std::isfinite(NativePixelChannel(pix+i*bpp,format,3)))return false;
 for(size_t i=0;i<n;i++)for(unsigned c=0;c<3;c++){
  const unsigned channel=format==NativePixelFormat::bgra8_unorm?2-c:c;
  float v=rgb[i*3+c];
  if(format!=NativePixelFormat::rgba16_float)v=v<0.f?0.f:v>1.f?1.f:v;
  if(bpp==4)pix[i*bpp+channel]=uint8_t(v*255.f+0.5f);
  else{
   uint16_t h=format==NativePixelFormat::rgba16_float?NativeFloatToHalf(v):uint16_t(v*65535.f+0.5f);
   pix[i*bpp+c*2]=uint8_t(h);pix[i*bpp+c*2+1]=uint8_t(h>>8);
  }
 }
 return true;
}

enum class NativeFrameInputCheck { valid, wrong_size, nonfinite, black };
inline NativeFrameInputCheck CheckNativeFrameInput(const std::vector<unsigned char>&bytes,NativePixelFormat format,unsigned width,unsigned height){
 const size_t bpp=NativePixelBytes(format);
 if(!bpp||!NativePixelGeometry(width,height)||bytes.size()!=size_t(width)*height*bpp)return NativeFrameInputCheck::wrong_size;
 bool rgb=false;
 for(size_t i=0;i<bytes.size();i+=bpp)for(unsigned c=0;c<4;c++){
  const float v=NativePixelChannel(bytes.data()+i,format,c);
  if(!std::isfinite(v))return NativeFrameInputCheck::nonfinite;
  if(c!=3&&v!=0.f)rgb=true;
 }
 return rgb?NativeFrameInputCheck::valid:NativeFrameInputCheck::black;
}
inline bool NativeFrameRequestValid(uint32_t current_pid,uint32_t request_pid,uint32_t request,uint32_t last){
 return current_pid&&current_pid==request_pid&&request>last&&request<=1000000;
}
// Compatibility for existing diagnostics ONLY: RGBA8 or FP16 by byte count.
// Cannot identify UNORM16. New texture callers must use the explicit overload.
inline NativeFrameInputCheck CheckNativeFrameInput(const std::vector<unsigned char>&bytes,unsigned width=1920,unsigned height=1080){
 if(!width||!height||width>7680||height>4320)return NativeFrameInputCheck::wrong_size;
 const size_t pixels=size_t(width)*height;
 /* 8-bit UNORM frames (Magpie, 4 bytes per pixel): no NaN possible; black = every RGB byte zero */
 if(bytes.size()==pixels*4){for(size_t i=0;i<bytes.size();i++)if(i%4!=3&&bytes[i])return NativeFrameInputCheck::valid;return NativeFrameInputCheck::black;}
 if(bytes.size()!=pixels*8)return NativeFrameInputCheck::wrong_size;
 bool rgb=false;
 for(size_t i=0;i<bytes.size();i+=2){uint16_t h;std::memcpy(&h,bytes.data()+i,2);
  if((h&0x7c00)==0x7c00)return NativeFrameInputCheck::nonfinite;
  if((i/2)%4!=3&&(h&0x7fff))rgb=true;
 }
 return rgb?NativeFrameInputCheck::valid:NativeFrameInputCheck::black;
}
