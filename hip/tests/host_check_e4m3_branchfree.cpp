#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
using u32=uint32_t; using u8=uint8_t;
static inline u32 as_u32(float v){u32 u;memcpy(&u,&v,4);return u;}
static inline float as_f32(u32 u){float v;memcpy(&v,&u,4);return v;}
// originals
static inline u8 e4m3_byte(float v){ u32 b=as_u32(v),a=b&0x7fffffffu; u8 sg=u8((b>>24)&0x80u);
 if(a>0x7f800000u) return u8(sg|0x7f); if(a>=0x43e00000u) return u8(sg|0x7e);
 if(a<0x3c800000u) return u8(sg|u8(nearbyintf(fabsf(v)*512.f)));
 u32 rounded=(a+0x7ffffu+((a>>20)&1u))&0xfff00000u; u32 code=((rounded>>23)-120u)*8u+((rounded>>20)&7u);
 return u8(sg|(code>126u?126u:code)); }
static inline float F_sw(float v){ u32 bits=as_u32(v),a=bits&0x7fffffffu; if(a>=0x7f800000u) return v;
 float sg=v<0?-1.f:1.f; if(a<0x3c800000u) return copysignf(nearbyintf(fabsf(v)*512.f)/512.f,v);
 if(a>=0x43e00000u) return sg*448.f; u32 r=(a+0x7ffffu+((a>>20)&1u))&0xfff00000u; float m=as_f32(r); return sg*(m>448.f?448.f:m); }
// branch-free
static inline u32 msk(bool c){return 0u-u32(c);}
static inline u8 e4m3_byte_nb(float v){ u32 b=as_u32(v),a=b&0x7fffffffu, sg=(b>>24)&0x80u;
 u32 q=u32(nearbyintf(fminf(fabsf(v),1.f)*512.f));
 u32 r=(a+0x7ffffu+((a>>20)&1u))&0xfff00000u;
 u32 cn=((r>>23)-120u)*8u+((r>>20)&7u); u32 mc=msk(cn>126u); cn=(cn&~mc)|(126u&mc);
 u32 ms=msk(a<0x3c800000u), mt=msk(a>=0x43e00000u), mn=msk(a>0x7f800000u);
 u32 code=(q&ms)|(cn&~ms); code=(code&~mt)|(126u&mt); code=(code&~mn)|(127u&mn);
 return u8(sg|code); }
static inline float F_nb(float v){ u32 bits=as_u32(v),a=bits&0x7fffffffu,s=bits&0x80000000u;
 u32 sub=as_u32(nearbyintf(fminf(fabsf(v),1.f)*512.f)/512.f);
 u32 r=(a+0x7ffffu+((a>>20)&1u))&0xfff00000u; u32 mr=msk(r>0x43e00000u); r=(r&~mr)|(0x43e00000u&mr);
 u32 ms=msk(a<0x3c800000u), mt=msk(a>=0x43e00000u), mi=msk(a>=0x7f800000u);
 u32 mag=(sub&ms)|(r&~ms); mag=(mag&~mt)|(0x43e00000u&mt);
 u32 out=s|mag; out=(out&~mi)|(bits&mi); return as_f32(out); }
int main(){ uint64_t bad1=0,bad2=0; u32 ex1=0,ex2=0;
 for(uint64_t i=0;i<=0xffffffffull;i++){ float v=as_f32(u32(i));
  if(e4m3_byte(v)!=e4m3_byte_nb(v)){ if(!bad1)ex1=u32(i); bad1++; }
  if(as_u32(F_sw(v))!=as_u32(F_nb(v))){ if(!bad2)ex2=u32(i); bad2++; } }
 printf("e4m3_byte mismatches=%llu (first %08x)\nF mismatches=%llu (first %08x)\n",(unsigned long long)bad1,ex1,(unsigned long long)bad2,ex2); }
