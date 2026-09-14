// Compile with the generated query-owned source included before this file.
// Standalone device regression: no production runtime dependency.
#include <cuda_runtime.h>
#include <cstdio>

__device__ __noinline__ float reference_weight(float score, float next) {
  return isinf(score) ? 0.0f : expf(score-next);
}
__device__ __noinline__ unsigned reference_pair(float low, float high) {
  return (unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(low)) |
    ((unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(high)) << 16);
}
__global__ void numeric_regression(unsigned *failures) {
  const unsigned lows[8]={0,1,0x7fff,0x8000,0x8001,0xfffe,0xffff,0x5555};
  for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<65536U*8U;i+=gridDim.x*blockDim.x) {
    // Every sign/exponent/upper-mantissa pattern plus rounding ties/neighbours.
    // Includes both infinities, signed zero, subnormals, quiet/signalling NaNs.
    const unsigned bits=((i/8)<<16)|lows[i%8];
    const float score=__uint_as_float(bits);
    const float other=__uint_as_float(bits^0x807fffffU);
    if(reference_pair(score,other)!=lf_probability_pair(score,other))
      atomicAdd(failures,1U);
    const float maxima[6]={0.0f,score,other,CUDART_INF_F,-CUDART_INF_F,__uint_as_float(0x7fc00001)};
    for(int j=0;j<6;++j) {
      const float reference=reference_weight(score,maxima[j]);
      const float result=lf_masked_exp(score,score-maxima[j]);
      if(__float_as_uint(reference)!=__float_as_uint(result)) atomicAdd(failures+1,1U);
    }
  }
}
int main() {
  unsigned *device=nullptr, failures[2]={};
  cudaError_t status=cudaMalloc(&device,sizeof(failures));
  if(status!=cudaSuccess) return 2;
  status=cudaMemset(device,0,sizeof(failures));
  if(status==cudaSuccess) {
    numeric_regression<<<128,128>>>(device);
    status=cudaGetLastError();
  }
  if(status==cudaSuccess) status=cudaDeviceSynchronize();
  if(status==cudaSuccess) status=cudaMemcpy(failures,device,sizeof(failures),cudaMemcpyDeviceToHost);
  const cudaError_t release=cudaFree(device);
  printf("pairs=524288 exponent_selections=3145728 pair_mismatches=%u selection_mismatches=%u status=%d release=%d\n",
    failures[0],failures[1],(int)status,(int)release);
  return status!=cudaSuccess || release!=cudaSuccess || failures[0] || failures[1];
}
