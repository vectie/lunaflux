#include <cuda_bf16.h>
extern "C" __global__ void lunaflux_luna_residual_add_v1_ep_7(const __nv_bfloat16 *left, const __nv_bfloat16 *right, __nv_bfloat16 *output) {
  const unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 16ULL) {
    output[index] = __hadd(left[index], right[index]);
  }
}
