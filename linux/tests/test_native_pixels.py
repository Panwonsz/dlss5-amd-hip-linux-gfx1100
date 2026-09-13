"""CPU-only regression harness; no Wine, GPU, or game access."""
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def run_cpp(source):
    with tempfile.TemporaryDirectory() as tmp:
        src = pathlib.Path(tmp) / 'test.cpp'
        src.write_text(source)
        exe = pathlib.Path(tmp) / 'test'
        build = subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                                '-I', str(ROOT / 'src'), str(src), '-o', str(exe)],
                               text=True, capture_output=True)
        if build.returncode:
            raise AssertionError(build.stdout + build.stderr)
        run = subprocess.run([str(exe)], text=True, capture_output=True)
        if run.returncode:
            raise AssertionError(run.stdout + run.stderr)


class PixelTests(unittest.TestCase):
    def test_half_roundtrips_and_all_finite_midpoints(self):
        # Initially exercise the existing implementation without its Windows loader.
        header = (ROOT / 'src/native_frame_input_check.h').read_text()
        legacy = (ROOT / 'src/native_hip_client.h').read_text()
        implementation = ''
        if 'inline uint16_t NativeFloatToHalf' not in header:
            implementation = legacy.split('inline uint16_t NativeFloatToHalf', 1)[1].split(
                'inline constexpr unsigned NativeHipNetWidth', 1)[0]
            implementation = 'inline uint16_t NativeFloatToHalf' + implementation
        run_cpp('''
#include "native_frame_input_check.h"
#include <cmath>
#include <cstdio>
#include <limits>
''' + implementation + r'''
// Independent exact mathematical half decoder (not production bit conversion).
float reference(unsigned h) {
    unsigned e=(h>>10)&31, m=h&1023;
    float v=e==0 ? std::ldexp(float(m),-24) :
        e==31 ? (m ? std::numeric_limits<float>::quiet_NaN() : INFINITY) :
        std::ldexp(float(1024+m),int(e)-25);
    return h&0x8000 ? -v : v;
}
int main() {
    for(unsigned h=0;h<65536;h++) {
        unsigned got=NativeFloatToHalf(reference(h));
        bool nan=(h&0x7c00)==0x7c00 && (h&1023);
        if(nan ? ((got&0x7c00)!=0x7c00 || !(got&1023) || ((got^h)&0x8000)) : got!=h) {
            std::fprintf(stderr,"roundtrip %04x -> %04x\n",h,got);return 1;
        }
    }
    for(unsigned h=0;h<0x7bff;h++) {
        float a=reference(h),b=reference(h+1),mid=(a+b)*0.5f;
        unsigned even=h+(h&1);
        for(unsigned sign: {0u,0x8000u}) {
            float s=sign ? -1.f : 1.f;
            if(NativeFloatToHalf(s*mid)!=(sign|even) ||
               NativeFloatToHalf(s*std::nextafter(mid,a))!=(sign|h) ||
               NativeFloatToHalf(s*std::nextafter(mid,b))!=(sign|(h+1))) {
                std::fprintf(stderr,"midpoint %04x sign=%04x\n",h,sign);return 2;
            }
        }
    }
    for(float s: {1.f,-1.f}) {
        unsigned sign=s<0 ? 0x8000 : 0;
        if(NativeFloatToHalf(s*65520.f)!=(sign|0x7c00) ||
           NativeFloatToHalf(s*std::nextafter(65520.f,0.f))!=(sign|0x7bff))return 3;
    }
}
''')

    def test_explicit_formats_alpha_and_atomic_rejection(self):
        run_cpp(r'''
#include "native_frame_input_check.h"
#include <cassert>
#include <cmath>
#include <limits>
int main() {
    using F=NativePixelFormat;
    std::vector<float> rgba;
    unsigned char p8[]={20,40,80,51};
    assert(NativePixelsToRgbaF32(p8,F::bgra8_unorm,1,1,rgba));
    assert(rgba[0]==80/255.f && rgba[2]==20/255.f && rgba[3]==51/255.f);
    assert(NativePixelsToRgbaF32(p8,F::rgba8_unorm,1,1,rgba));
    assert(rgba[0]==20/255.f && rgba[2]==80/255.f && rgba[3]==51/255.f);
    // sRGB is explicitly unsupported: no silent transfer-function guessing.
    assert(NativePixelFormatFromDxgi(29)==F::unsupported);
    assert(NativePixelFormatFromDxgi(91)==F::unsupported);
    assert(NativePixelFormatFromDxgi(27)==F::unsupported);
    assert(NativePixelFormatFromDxgi(28)==F::rgba8_unorm);
    assert(NativePixelFormatFromDxgi(87)==F::bgra8_unorm);
    assert(NativePixelFormatFromDxgi(10)==F::rgba16_float);
    assert(NativePixelFormatFromDxgi(11)==F::rgba16_unorm);
    uint16_t words[]={0x3800,0x7c00,0xffff,0x1234};
    auto *p16=reinterpret_cast<unsigned char*>(words);
    assert(NativePixelsToRgbaF32(p16,F::rgba16_unorm,1,1,rgba));
    assert(rgba[0]==0x3800/65535.f && rgba[1]==0x7c00/65535.f && rgba[2]==1.f && rgba[3]==0x1234/65535.f);
    words[0]=NativeFloatToHalf(4.f);words[1]=NativeFloatToHalf(-0.5f);
    words[2]=NativeFloatToHalf(0.25f);words[3]=NativeFloatToHalf(0.5f);
    assert(NativePixelsToRgbaF32(p16,F::rgba16_float,1,1,rgba));
    assert(rgba[0]==4.f && rgba[1]==-0.5f && rgba[2]==0.25f && rgba[3]==0.5f);
    std::vector<unsigned char> bytes(p16,p16+8);
    assert(CheckNativeFrameInput(bytes,F::rgba16_float,1,1)==NativeFrameInputCheck::valid);
    words[0]=0x7c00;
    bytes.assign(p16,p16+8);
    assert(CheckNativeFrameInput(bytes,F::rgba16_float,1,1)==NativeFrameInputCheck::nonfinite);
    assert(CheckNativeFrameInput(bytes,F::rgba16_unorm,1,1)==NativeFrameInputCheck::valid);
    auto saved=rgba;
    assert(!NativePixelsToRgbaF32(p16,F::rgba16_float,1,1,rgba) && rgba==saved);
    float rgb[]={1.f,0.5f,0.f};
    assert(NativeRgbF32ToPixels(rgb,p8,F::bgra8_unorm,1,1));
    assert(p8[0]==0 && p8[1]==128 && p8[2]==255 && p8[3]==51);
    assert(NativeRgbF32ToPixels(rgb,p8,F::rgba8_unorm,1,1));
    assert(p8[0]==255 && p8[2]==0 && p8[3]==51);
    assert(NativeRgbF32ToPixels(rgb,p16,F::rgba16_unorm,1,1));
    assert(words[0]==65535 && words[1]==32768 && words[2]==0 && words[3]==0x3800);
    rgb[0]=4.f;rgb[1]=-0.5f;
    assert(NativeRgbF32ToPixels(rgb,p16,F::rgba16_float,1,1));
    assert(words[0]==NativeFloatToHalf(4.f) && words[1]==NativeFloatToHalf(-0.5f) && words[3]==0x3800);
    for(F f:{F::rgba8_unorm,F::bgra8_unorm,F::rgba16_unorm,F::rgba16_float}) {
        std::vector<unsigned char> dest(2*NativePixelBytes(f),42),before=dest;
        float bad[]={0.1f,0.2f,0.3f,0.4f,0.5f,INFINITY};
        for(float v:{INFINITY,-INFINITY,std::numeric_limits<float>::quiet_NaN()}) {
            bad[5]=v;
            assert(!NativeRgbF32ToPixels(bad,dest.data(),f,2,1) && dest==before);
        }
    }
    auto before=bytes;
    rgb[0]=65520.f;
    assert(!NativeRgbF32ToPixels(rgb,bytes.data(),F::rgba16_float,1,1) && bytes==before);
    assert(!NativePixelsToRgbaF32(nullptr,F::rgba8_unorm,1,1,rgba));
    assert(!NativePixelsToRgbaF32(p8,F::unsupported,1,1,rgba));
    assert(!NativePixelsToRgbaF32(p8,F::rgba8_unorm,~0u,~0u,rgba));
    assert(!NativeRgbF32ToPixels(rgb,p8,F::rgba8_unorm,0,1));
    bytes.assign(8,0);bytes[6]=0xff;bytes[7]=0xff;
    assert(CheckNativeFrameInput(bytes,F::rgba16_unorm,1,1)==NativeFrameInputCheck::black);
    assert(CheckNativeFrameInput(bytes,F::rgba16_float,1,1)==NativeFrameInputCheck::nonfinite);
    assert(CheckNativeFrameInput(bytes,F::rgba8_unorm,1,1)==NativeFrameInputCheck::wrong_size);
}
''')


if __name__ == '__main__':
    unittest.main()
