// Full-width rotary + simulated KV precision. Independent CPU oracle; not
// checkpoint inference or a performance benchmark. All allocations are tiny.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "rotary_kv_kernels.cuh"

static void check(cudaError_t error) {
  if (error != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(error)); std::exit(2); }
}
template<class T> static T *allocate(size_t count) {
  T *result = nullptr;
  check(cudaMallocManaged(&result, count * sizeof(T)));
  std::memset(result, 0, count * sizeof(T));
  return result;
}
static float bf(float value) { return __bfloat162float(__float2bfloat16_rn(value)); }
static unsigned short word(float value) {
  __nv_bfloat16 stored=__float2bfloat16_rn(value);
  unsigned short result; std::memcpy(&result,&stored,2); return result;
}
static float frequency(int pair, bool yarn) {
  const float theta=yarn?160000.f:10000.f;
  const float base=1.f/std::pow(theta,float(pair*2)/64.f);
  if (!yarn) return base;
  const float low=std::max(0.f,float(std::floor(64.0*std::log(65536.0/(32.0*2.0*3.141592653589793))/(2.0*std::log(160000.0)))));
  const float high=std::min(63.f,float(std::ceil(64.0*std::log(65536.0/(2.0*3.141592653589793))/(2.0*std::log(160000.0)))));
  const float smooth=1.f-std::max(0.f,std::min(1.f,(float(pair)-low)/(high-low+(high==low?0.001f:0.f))));
  return base/16.f*(1.f-smooth)+base*smooth;
}
static void rotate(const __nv_bfloat16 *input, float *output, int heads,
                   const int *positions, int live, bool yarn) {
  for (int row=0;row<live;++row) for (int head=0;head<heads;++head) {
    const int base=(row*heads+head)*512;
    for (int col=0;col<448;++col) output[base+col]=__bfloat162float(input[base+col]);
    for (int pair=0;pair<32;++pair) {
      const float angle=float(positions[row])*frequency(pair,yarn);
      const float cosine=std::cos(angle), sine=std::sin(angle);
      const int index=base+448+pair*2;
      const float l=__bfloat162float(input[index]), r=__bfloat162float(input[index+1]);
      output[index]=bf(l*cosine-r*sine);
      output[index+1]=bf(l*sine+r*cosine);
    }
  }
}
static void simulate(float *values, int live, bool rounded) {
  for (int row=0;row<live;++row) for (int group=0;group<7;++group) {
    const int base=row*512+group*64;
    float maximum=0.0001f;
    for (int col=0;col<64;++col) maximum=std::max(maximum,std::fabs(values[base+col]));
    float scale=maximum*(1.f/448.f);
    if (rounded) {
      int exponent=0; const float mantissa=std::frexp(scale,&exponent);
      scale=std::ldexp(1.f, mantissa==0.5f?exponent-1:exponent);
    }
    for (int col=0;col<64;++col) {
      const float bounded=std::max(-448.f,std::min(448.f,values[base+col]/scale));
      __nv_fp8_e4m3 value; value.__x=__nv_cvt_float_to_fp8(bounded,__NV_SATFINITE,__NV_E4M3);
      values[base+col]=bf(float(value)*scale);
    }
  }
}
int main() {
  constexpr int R=3,Q=2,D=512;
  auto counts=allocate<int>(5), positions=allocate<int>(R);
  auto qi=allocate<__nv_bfloat16>(R*Q*D), ki=allocate<__nv_bfloat16>(R*D);
  auto qo=allocate<unsigned short>(R*Q*D), ko=allocate<unsigned short>(R*D);
  std::vector<float> qr(R*Q*D), kr(R*D);
  for (int mode : {0,1,2}) for (int zero : {0,1}) for (int live : {0,1,3}) {
    counts[3]=live; positions[0]=0; positions[1]=65536; positions[2]=1048575;
    for (int i=0;i<R*Q*D;++i) qi[i]=__float2bfloat16_rn(zero?0.f:float((i*13)%31-15)*0.125f);
    for (int i=0;i<R*D;++i) ki[i]=__float2bfloat16_rn(zero?0.f:float((i*7)%37-18)*0.0625f);
    rotate(qi,qr.data(),Q,positions,live,mode==1);
    rotate(ki,kr.data(),1,positions,live,mode==1);
    simulate(kr.data(),live,mode!=2);
    std::vector<unsigned short> first_q,first_k;
    for (int repeat=0;repeat<2;++repeat) {
      std::fill(qo,qo+R*Q*D,word(19.f)); std::fill(ko,ko+R*D,word(19.f));
      if (mode==0) {
        rotary_fixture_rotary<<<dim3(R,Q+1),256>>>(counts,positions,(unsigned short *)qi,(unsigned short *)ki,qo,ko);
        rotary_fixture_simulate<<<dim3(R,7),64>>>(counts,(__nv_bfloat16 *)ko);
      } else if (mode==1) {
        yarn_fixture_rotary<<<dim3(R,Q+1),256>>>(counts,positions,(unsigned short *)qi,(unsigned short *)ki,qo,ko);
        yarn_fixture_simulate<<<dim3(R,7),64>>>(counts,(__nv_bfloat16 *)ko);
      } else {
        linear_fixture_rotary<<<dim3(R,Q+1),256>>>(counts,positions,(unsigned short *)qi,(unsigned short *)ki,qo,ko);
        linear_fixture_simulate<<<dim3(R,7),64>>>(counts,(__nv_bfloat16 *)ko);
      }
      check(cudaGetLastError()); check(cudaDeviceSynchronize());
      float maximum=0.f;
      for (int i=0;i<live*Q*D;++i) {
        maximum=std::max(maximum,std::fabs(__bfloat162float(((__nv_bfloat16 *)qo)[i])-qr[i]));
        if (i%D<448 && qo[i]!=word(__bfloat162float(qi[i]))) return 4;
      }
      for (int i=0;i<live*D;++i) {
        maximum=std::max(maximum,std::fabs(__bfloat162float(((__nv_bfloat16 *)ko)[i])-kr[i]));
        if (i%D<448 && ko[i]!=word(kr[i])) { std::fprintf(stderr,"quant mode=%d index=%d\n",mode,i); return 5; }
      }
      if (maximum>0.03125f) { std::fprintf(stderr,"mode=%d maxabs=%.9g\n",mode,maximum); return 3; }
      for (int i=live*Q*D;i<R*Q*D;++i) if (qo[i]!=word(19.f)) return 6;
      for (int i=live*D;i<R*D;++i) if (ko[i]!=word(19.f)) return 7;
      if (repeat==0) { first_q.assign(qo,qo+R*Q*D); first_k.assign(ko,ko+R*D); }
      else if (std::memcmp(first_q.data(),qo,R*Q*D*2)||std::memcmp(first_k.data(),ko,R*D*2)) return 8;
      std::printf("mode=%d zero=%d live=%d repeat=%d maxabs=%.9g prefix=exact quant=exact inactive=untouched\n",mode,zero,live,repeat,maximum);
    }
  }
  // Bad rotary positions must leave that complete row unpublished.
  counts[3]=1; positions[0]=-1; std::fill(qo,qo+R*Q*D,word(19.f)); std::fill(ko,ko+R*D,word(19.f));
  rotary_fixture_rotary<<<dim3(R,Q+1),256>>>(counts,positions,(unsigned short *)qi,(unsigned short *)ki,qo,ko);
  check(cudaGetLastError()); check(cudaDeviceSynchronize());
  for (int i=0;i<R*Q*D;++i) if (qo[i]!=word(19.f)) return 9;
  for (int i=0;i<R*D;++i) if (ko[i]!=word(19.f)) return 10;
  check(cudaFree(ko)); check(cudaFree(qo)); check(cudaFree(ki)); check(cudaFree(qi));
  check(cudaFree(positions)); check(cudaFree(counts));
  std::puts("rotary_kv=passed deterministic=true cache_simulation=bf16 all_allocations=released");
}
