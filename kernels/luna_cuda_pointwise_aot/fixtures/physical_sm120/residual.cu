#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
extern "C" __global__ void lunaflux_luna_residual_add_v1_op_2_ep_3(const int32_t *counts, const __nv_bfloat16 *left, const __nv_bfloat16 *right, __nv_bfloat16 *output) {
  const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;
  const int live_tokens = counts[3];
  if (live_tokens <= 0) return;
  const unsigned long long live = (unsigned long long)live_tokens * 8ULL;
  if (i < live) {
    const float sum = __bfloat162float(left[i]) + __bfloat162float(right[i]);
    output[i] = __float2bfloat16_rn(sum);
  }
}
