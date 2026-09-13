"""CPU-only tests of the real client header; only platform/C-API edges are stubbed.

The header is copied verbatim into a temporary include tree so its unrelated
D3D lab-path dependency can be replaced without modifying production sources.
No Wine, GPU, vendor library, weights, or game is loaded.
"""
import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
WINDOWS = r'''
#pragma once
#include <cstdio>
#include <cwchar>
#include <string>
using HMODULE=void*;
using LPCWSTR=const wchar_t*;
constexpr int MAX_PATH=260;
constexpr int GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS=1;
constexpr int GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT=2;
inline int GetModuleHandleExW(int,LPCWSTR,HMODULE*h){*h=(void*)1;return 1;}
inline unsigned GetModuleFileNameW(HMODULE,wchar_t*p,int){std::wcscpy(p,L"/stub/addon.dll");return 15;}
inline unsigned long GetCurrentProcessId(){return 1;}
inline unsigned long long GetTickCount64(){return 1;}
inline FILE* _wfopen(const wchar_t*,const wchar_t*){return nullptr;}
inline const wchar_t* _wgetenv(const wchar_t*){return nullptr;}
HMODULE LoadLibraryW(const wchar_t*);
int FreeLibrary(HMODULE);
using FARPROC=void(*)();
FARPROC GetProcAddress(HMODULE,const char*);
'''
STUBS = r'''
#include "native_hip_client.h"
#include <cassert>
#include <atomic>
#include <thread>
#include <type_traits>
#include <vector>
static std::atomic<int> loads{0}, frees{0}, inits{0}, shuts{0}, probes{0}, runs{0};
static std::atomic<bool> model{false};
static bool fail_init=false, fail_load=false, v2=true;
static int probe_rc=-1;
static const char* probe_error="invalid frame ABI/buffers";
static const char* missing="";
static thread_local const char* error="";
static int init_gpu=-1;
static int init(const char*,int gpu){init_gpu=gpu;++inits;if(fail_init){error="init rejected";return -1;}model=true;return 0;}
static int selected_gpu=-1;
static int find_device(const char* name){assert(!std::strcmp(name,"exact adapter"));selected_gpu=1;return 1;}
static int run(const float*,float*,unsigned){++runs;return 0;}
static int frame(const Dlss5Frame*f){
 if(!f){++probes;error=v2 ? (model ? probe_error : "not initialized") :
 "hip frame bridge missing (rebuild and preload the matching V2 library)";return probe_rc;}
 ++runs;assert(model);return 0;
}
static const char* err(){return error;}
static void shut(){++shuts;model=false;}
HMODULE LoadLibraryW(const wchar_t*){++loads;return fail_load?nullptr:(void*)2;}
int FreeLibrary(HMODULE){++frees;return 1;}
FARPROC GetProcAddress(HMODULE,const char*n){
 if(!std::strcmp(n,missing))return nullptr;
 if(!std::strcmp(n,"dlss5_init"))return reinterpret_cast<FARPROC>(&init);
 if(!std::strcmp(n,"dlss5_find_device"))return reinterpret_cast<FARPROC>(&find_device);
 if(!std::strcmp(n,"dlss5_run"))return reinterpret_cast<FARPROC>(&run);
 if(!std::strcmp(n,"dlss5_run_frame"))return reinterpret_cast<FARPROC>(&frame);
 if(!std::strcmp(n,"dlss5_last_error"))return reinterpret_cast<FARPROC>(&err);
 if(!std::strcmp(n,"dlss5_shutdown"))return reinterpret_cast<FARPROC>(&shut);
 return nullptr;
}
static bool rejected(NativeHipClient&c){try{c.Create();return false;}catch(const std::runtime_error&){return true;}}
'''


def run_cpp(body):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        (tmp / 'src').mkdir()
        (tmp / 'hip/include').mkdir(parents=True)
        for name in ('native_hip_client.h', 'native_frame_input_check.h'):
            shutil.copyfile(ROOT / 'src' / name, tmp / 'src' / name)
        shutil.copyfile(ROOT / 'hip/include/dlss5_capi.h', tmp / 'hip/include/dlss5_capi.h')
        (tmp / 'src/windows.h').write_text(WINDOWS)
        (tmp / 'src/native_lab_paths.h').write_text(
            '#pragma once\n#include <string>\n#include <mutex>\n'
            'inline std::wstring NativeLabRoot(){return L"/stub";}\n'
            'inline std::wstring NativeLabPath(const wchar_t*p){return p;}\n')
        (tmp / 'test.cpp').write_text(STUBS + '\nint main(){\n' + body + '\n}\n')
        command = ['g++', '-std=c++17', '-O0', '-Wall', '-Wextra', '-pthread',
                   '-I', str(tmp / 'src'), str(tmp / 'test.cpp'), '-o', str(tmp / 'test')]
        build = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if build.returncode:
            raise AssertionError('compile failed:\n' + build.stdout + build.stderr)
        result = subprocess.run([str(tmp / 'test')], capture_output=True, text=True, timeout=10)
        if result.returncode:
            raise AssertionError('client harness failed:\n' + result.stdout + result.stderr)


class NativeHipClientTests(unittest.TestCase):
    def test_explicit_adapter_uses_hip_lookup(self):
        run_cpp(r'''
NativeHipClient c;c.Create("exact adapter");assert(c.Ready());
assert(selected_gpu==1 && init_gpu==1);
''')
    def test_unknown_or_successful_null_probe_fails_closed(self):
        run_cpp(r'''
for(const char*message:{"", "not initialized", "unknown bridge", static_cast<const char*>(nullptr)}){
 probe_error=message;
 {NativeHipClient c;assert(rejected(c));assert(!c.Ready());}
 assert(!model && loads==frees && inits==shuts);
}
probe_error="invalid frame ABI/buffers";probe_rc=0;
{NativeHipClient c;assert(rejected(c));assert(!c.Ready());}
assert(!model && runs==0 && loads==frees && inits==shuts);
''')

    def test_v2_handshake_before_ready_and_cleanup_after_rejection(self):
        run_cpp(r'''
v2=false;
{
 NativeHipClient c;assert(rejected(c));assert(!c.Ready());
 assert(inits==1 && probes==1 && runs==0);
 assert(shuts==1 && frees==1 && !model);
 assert(c.RunFrame(nullptr,nullptr,0,false)==-1);
}
assert(shuts==1);v2=true;
{NativeHipClient c;c.Create();assert(c.Ready());assert(probes==2 && runs==0);}
assert(shuts==2 && frees==2);
''')

    def test_successful_handshake_and_frame_forwarding(self):
        run_cpp(r'''
NativeHipClient c;c.Create();assert(c.Ready());
assert(probes==1 && runs==0 && inits==1 && shuts==0);
float input=0,output=0;
assert(c.Run(&input,&output,4)==0);
assert(c.RunFrame(&input,&output,7,true)==0);
assert(runs==2 && probes==1);
''')

    def test_every_used_export_is_required(self):
        run_cpp(r'''
for(const char*name:{"dlss5_init","dlss5_run","dlss5_run_frame","dlss5_last_error","dlss5_shutdown"}){
 missing=name;model=true;int before=inits, cleanup=frees;
 {NativeHipClient c;assert(rejected(c));assert(!c.Ready());
 assert(inits==before && shuts==0 && model && frees==cleanup+1);}
}
missing="";{NativeHipClient c;c.Create();assert(c.Ready());}
assert(shuts==1 && loads==frees);
''')

    def test_load_failure_releases_reservation(self):
        run_cpp(r'''
fail_load=true;NativeHipClient a;assert(rejected(a));
assert(!a.Ready() && inits==0 && shuts==0 && frees==0);
fail_load=false;{NativeHipClient b;b.Create();assert(b.Ready());}
assert(shuts==1 && frees==1);
''')

    def test_single_owner_repeated_create_and_release(self):
        run_cpp(r'''
{
 NativeHipClient a;a.Create();assert(a.Ready());
 assert(rejected(a));assert(a.Ready());assert(loads==1 && inits==1);
 {NativeHipClient b;assert(rejected(b));assert(!b.Ready());}
 assert(model && shuts==0 && loads==1);
}
assert(!model && shuts==1 && frees==1);
{NativeHipClient c;c.Create();assert(c.Ready());}
assert(shuts==2 && frees==2);
''')

    def test_noncopyable_nonmovable(self):
        run_cpp(r'''
assert(!std::is_copy_constructible<NativeHipClient>::value);
assert(!std::is_copy_assignable<NativeHipClient>::value);
assert(!std::is_move_constructible<NativeHipClient>::value);
assert(!std::is_move_assignable<NativeHipClient>::value);
''')

    def test_failed_init_retries_without_retained_modules(self):
        run_cpp(r'''
NativeHipClient a;fail_init=true;model=true;
assert(rejected(a));assert(frees==1 && shuts==0 && model);
assert(rejected(a));assert(frees==2 && shuts==0 && model);
fail_init=false;
{NativeHipClient b;b.Create();assert(b.Ready());assert(rejected(a));}
a.Create();assert(a.Ready());
assert(loads==4 && frees==3 && inits==4);
''')

    def test_concurrent_instances_have_one_owner(self):
        run_cpp(r'''
std::atomic<int> arrived{0},success{0};std::atomic<bool> go{false};
std::vector<std::thread> threads;
for(int i=0;i<12;i++)threads.emplace_back([&]{
 NativeHipClient c;while(!go.load())std::this_thread::yield();
 if(!rejected(c))++success;
 ++arrived;while(arrived.load()!=12)std::this_thread::yield();
});
go=true;for(auto&t:threads)t.join();
assert(success==1 && inits==1 && loads==1 && shuts==1 && frees==1);
''')

    def test_failed_init_preserves_external_model(self):
        run_cpp(r'''
model=true;fail_init=true;
{
 NativeHipClient c;assert(rejected(c));assert(!c.Ready());
 assert(inits==1 && model && shuts==0);
 assert(loads==1 && frees==1);
 assert(c.Run(nullptr,nullptr,0)==-1);
 assert(c.RunFrame(nullptr,nullptr,0,false)==-1);
}
assert(model && shuts==0 && frees==1 && probes==0 && runs==0);
''')


if __name__ == '__main__':
    unittest.main()
