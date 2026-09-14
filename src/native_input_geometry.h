#pragma once
#include <cstdint>
#include <stdexcept>

// External window pixels and network pixels are separate coordinate systems.
// Integer viewport dimensions keep padding boundaries on pixel edges.
struct NativeInputGeometry {
 unsigned width{},height{},x{},y{},fit_width{},fit_height{};
 static bool Supported(uint64_t w,unsigned h){return w>0&&h>0&&w<=1920&&h<=1080;}
 static NativeInputGeometry Make(unsigned w,unsigned h){
  if(!Supported(w,h))throw std::runtime_error("input must fit within 1920x1080");
  NativeInputGeometry g{w,h,0,0,1920,1080};
  if(uint64_t(w)*1080>=uint64_t(h)*1920)g.fit_height=unsigned((uint64_t(h)*1920+w/2)/w);
  else g.fit_width=unsigned((uint64_t(w)*1080+h/2)/h);
  if(!g.fit_width)g.fit_width=1;if(!g.fit_height)g.fit_height=1;
  g.x=(1920-g.fit_width)/2;g.y=(1080-g.fit_height)/2;return g;
 }
 bool Adapted()const{return width!=1920||height!=1080;}
 unsigned RowPitch(unsigned bytes_per_pixel)const{return (width*bytes_per_pixel+255u)&~255u;}
};
