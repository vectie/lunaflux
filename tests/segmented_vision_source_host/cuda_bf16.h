#pragma once
// Test-only host model of the CUDA spelling used by the emitted source.
// It is neither a CUDA runtime nor evidence of device execution.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <condition_variable>
#define __global__
#define __shared__ static
struct __nv_bfloat16 { uint16_t bits; };
inline __nv_bfloat16 __float2bfloat16_rn(float value) {
  uint32_t bits; std::memcpy(&bits, &value, 4);
  if ((bits & 0x7fffffffU) > 0x7f800000U) return {uint16_t((bits >> 16) | 0x40)};
  return {uint16_t((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16)};
}
inline float __bfloat162float(__nv_bfloat16 value) {
  uint32_t bits = uint32_t(value.bits) << 16; float out; std::memcpy(&out, &bits, 4); return out;
}
inline float __fadd_rn(float a,float b) { volatile float out=a+b; return out; }
inline float __fmul_rn(float a,float b) { volatile float out=a*b; return out; }
inline float __fdiv_rn(float a,float b) { volatile float out=a/b; return out; }
inline float __fmaf_rn(float a,float b,float c) { return std::fma(a,b,c); }
inline float rsqrtf(float x) { return 1.0f/std::sqrt(x); }
struct Dim { int x=0,y=0,z=0; };
inline thread_local Dim threadIdx, blockIdx, blockDim;
struct HostBarrier {
  std::mutex mutex;
  std::condition_variable condition;
  unsigned remaining=128, generation=0;
  void wait() {
    std::unique_lock<std::mutex> lock(mutex);
    unsigned current=generation;
    if (--remaining==0) { remaining=128; ++generation; condition.notify_all(); }
    else condition.wait(lock,[&]{ return generation!=current; });
  }
};
inline HostBarrier host_barrier;
inline void __syncthreads() { host_barrier.wait(); }
