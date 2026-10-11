// Bounded independent F64 oracle; actual production source is in main.cuh.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include "main.cuh"

static void check(cudaError_t result) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(result)); std::exit(2);
  }
}
template<class T> static T *allocate(size_t count) {
  T *result=nullptr; check(cudaMallocManaged(&result,count*sizeof(T))); return result;
}
static float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
static unsigned bits(__nv_bfloat16 x) {
  unsigned short raw; std::memcpy(&raw,&x,2); return raw;
}
static void fixture(__nv_bfloat16 *x,__nv_bfloat16 *w,int width,int heads,int pattern) {
  for (int i=0;i<width;++i) w[i]=__float2bfloat16_rn(0.5f+float(i%129)/128.f);
  for (int row=0;row<3*heads;++row) for (int d=0;d<width;++d) {
    float v=pattern==0?0.f:pattern==1?1.f:pattern==2?(d==0?4096.f:1.f):float((d*17+row*13)%101-50)/32.f;
    x[row*width+d]=__float2bfloat16_rn(v);
  }
}
static std::vector<__nv_bfloat16> oracle(const __nv_bfloat16 *x,const __nv_bfloat16 *w,int width,int heads) {
  std::vector<__nv_bfloat16> result(3*heads*width);
  for (int row=0;row<3*heads;++row) {
    double sum=0; for(int d=0;d<width;++d) { double v=__bfloat162float(x[row*width+d]); sum+=v*v; }
    float inverse=float(1.0/std::sqrt(sum/width+1.e-6));
    for(int d=0;d<width;++d) result[row*width+d]=__float2bfloat16_rn(bf(__bfloat162float(x[row*width+d])*inverse)*__bfloat162float(w[d]));
  }
  return result;
}
static void compare(const __nv_bfloat16 *actual,const std::vector<__nv_bfloat16>& expected) {
  for(size_t i=0;i<expected.size();++i) {
    unsigned a=bits(actual[i]),b=bits(expected[i]);
    // One BF16 step allows the final F32 reciprocal-square-root rounding only.
    if(!std::isfinite(__bfloat162float(actual[i])) || (a>b?a-b:b-a)>1) {
      std::fprintf(stderr,"norm index=%zu actual=%g oracle=%g\n",i,__bfloat162float(actual[i]),__bfloat162float(expected[i])); std::exit(3);
    }
  }
}
// F64 is deliberately not the oracle for F32 overflow: this numerical law
// retains the F32 square/sum domain, including a zero inverse after overflow.
static void overflow_fixture(__nv_bfloat16 *x,__nv_bfloat16 *w,int width,int heads,int exponent) {
  for(int d=0;d<width;++d) w[d]=__float2bfloat16_rn(1.f);
  for(int i=0;i<3*heads*width;++i) x[i]=__float2bfloat16_rn(std::ldexp(1.f,exponent));
}
static void compare_overflow(const __nv_bfloat16 *actual,size_t count,const char *name) {
  for(size_t i=0;i<count;++i) if(__bfloat162float(actual[i])!=0.f) {
    std::fprintf(stderr,"F32 overflow %s index=%zu actual=%g expected=0\n",name,i,__bfloat162float(actual[i])); std::exit(5);
  }
}
static void nonfinite_fixture(__nv_bfloat16 *x,__nv_bfloat16 *w,int width,int heads,bool nan) {
  overflow_fixture(x,w,width,heads,0);
  for(int row=0;row<3*heads;++row) x[row*width]=__float2bfloat16_rn(
    nan?std::numeric_limits<float>::quiet_NaN():std::numeric_limits<float>::infinity());
}
static void compare_nonfinite(const __nv_bfloat16 *actual,size_t count,int width,bool nan) {
  for(size_t i=0;i<count;++i) {
    const float v=__bfloat162float(actual[i]);
    const bool expected_nan=nan||i%width==0;
    if(expected_nan?!std::isnan(v):v!=0.f) {
      std::fprintf(stderr,"nonfinite propagation index=%zu actual=%g expected_nan=%d\n",i,v,expected_nan); std::exit(6);
    }
  }
}
int main() {
  auto x=allocate<__nv_bfloat16>(3*5120),w=allocate<__nv_bfloat16>(5120),y=allocate<__nv_bfloat16>(3*5120);
  auto q=allocate<__nv_bfloat16>(3*4*128),k=allocate<__nv_bfloat16>(3*2*128);
  auto qw=allocate<__nv_bfloat16>(128),kw=allocate<__nv_bfloat16>(128);
  std::vector<__nv_bfloat16> first(3*5120),first_q(3*4*128),first_k(3*2*128);
  int cases=0;
  for(int pattern=0;pattern<4;++pattern) for(int repeat=0;repeat<2;++repeat) {
    fixture(x,w,5120,1,pattern); fixture(q,qw,128,4,pattern); fixture(k,kw,128,2,pattern);
    auto ey=oracle(x,w,5120,1),eq=oracle(q,qw,128,4),ek=oracle(k,kw,128,2);
    compensated_norm<<<3,1>>>(x,w,y); compensated_qknorm<<<dim3(3,6),1>>>(q,k,qw,kw);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    compare(y,ey); compare(q,eq); compare(k,ek);
    if(repeat==0) {
      std::memcpy(first.data(),y,first.size()*2); std::memcpy(first_q.data(),q,first_q.size()*2); std::memcpy(first_k.data(),k,first_k.size()*2);
    } else if(std::memcmp(first.data(),y,first.size()*2) || std::memcmp(first_q.data(),q,first_q.size()*2) || std::memcmp(first_k.data(),k,first_k.size()*2)) return 4;
    ++cases;
  }
  for(int exponent : {61,80}) {
    // 2^61 has finite squares but an overflowing sum; 2^80 squares overflow.
    overflow_fixture(x,w,5120,1,exponent);
    overflow_fixture(q,qw,128,4,exponent);
    overflow_fixture(k,kw,128,2,exponent);
    ordered_norm<<<3,1>>>(x,w,y);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    compare_overflow(y,3*5120,"ordered-row");
    compensated_norm<<<3,1>>>(x,w,y);
    compensated_qknorm<<<dim3(3,6),1>>>(q,k,qw,kw);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    compare_overflow(y,3*5120,"compensated-row");
    compare_overflow(q,3*4*128,"compensated-query");
    compare_overflow(k,3*2*128,"compensated-key");
  }
  for(bool nan : {false,true}) {
    nonfinite_fixture(x,w,5120,1,nan);
    nonfinite_fixture(q,qw,128,4,nan);
    nonfinite_fixture(k,kw,128,2,nan);
    ordered_norm<<<3,1>>>(x,w,y);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    compare_nonfinite(y,3*5120,5120,nan);
    compensated_norm<<<3,1>>>(x,w,y);
    compensated_qknorm<<<dim3(3,6),1>>>(q,k,qw,kw);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    compare_nonfinite(y,3*5120,5120,nan);
    compare_nonfinite(q,3*4*128,128,nan);
    compare_nonfinite(k,3*2*128,128,nan);
  }
  fixture(x,w,5120,1,3);
  cudaEvent_t start,stop; check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
  for(int trial=0;trial<4;++trial) for(int route=0;route<2;++route) {
    const bool accurate=(trial%2==0?route:1-route)!=0;
    for(int i=0;i<5;++i) {
      if(accurate) compensated_norm<<<3,1>>>(x,w,y); else ordered_norm<<<3,1>>>(x,w,y);
    }
    check(cudaEventRecord(start));
    for(int i=0;i<100;++i) {
      if(accurate) compensated_norm<<<3,1>>>(x,w,y); else ordered_norm<<<3,1>>>(x,w,y);
    }
    check(cudaEventRecord(stop)); check(cudaEventSynchronize(stop));
    float milliseconds=0; check(cudaEventElapsedTime(&milliseconds,start,stop));
    std::printf("norm_trial=%d law=%s gpu_us=%.6f\n",trial,accurate?"compensated":"ordered",milliseconds*10.f);
  }
  check(cudaEventDestroy(stop)); check(cudaEventDestroy(start));
  check(cudaFree(kw)); check(cudaFree(qw)); check(cudaFree(k)); check(cudaFree(q));
  check(cudaFree(y)); check(cudaFree(w)); check(cudaFree(x)); check(cudaDeviceReset());
  std::printf("text normalization cases=%d f32_overflow_cases=2 nonfinite_propagation_cases=2 row_width=5120 head_width=128 deterministic=true f64_oracle_max_bf16_steps=1 allocations_released=true\n",cases);
}
