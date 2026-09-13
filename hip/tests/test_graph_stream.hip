#include "dlss5_launch.hpp"
#include "dlss5_runtime.hpp"
#include <thread>
#include <atomic>
#include <cstdio>
using namespace dlss5;
int main(){hipStream_t a{},b{};hip_check(hipStreamCreate(&a),"a");hip_check(hipStreamCreate(&b),"b");std::atomic<int> phase{0};bool good=false;
 std::thread first([&]{set_launch_stream(a);phase=1;while(phase.load()!=2)std::this_thread::yield();good=get_launch_stream()==a;set_launch_stream(nullptr);});
 std::thread second([&]{while(phase.load()!=1)std::this_thread::yield();set_launch_stream(b);phase=2;});first.join();second.join();hip_check(hipStreamDestroy(a),"destroy a");hip_check(hipStreamDestroy(b),"destroy b");std::printf("thread-local stream isolation=%s\n",good?"PASS":"FAIL");return good?0:1;}
