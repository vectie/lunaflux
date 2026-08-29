#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
extern "C" __global__ void lunaflux_luna_rms_norm_v1_op_1_ep_2(const int32_t *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *weight, __nv_bfloat16 *output) {
  const int row = (int)blockIdx.x;
  if (counts[3] <= 0 || row >= counts[3]) return;
  __shared__ float partial[256];
  float sum = 0.0f;
  const unsigned long long base = (unsigned long long)row * 8ULL;
  for (int column = threadIdx.x; column < 8; column += 256) {
    const float value = __bfloat162float(input[base + column]);
    sum = sum + value * value;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float inverse = rsqrtf(partial[0] / 8.0f + 0.00001f);
  for (int column = threadIdx.x; column < 8; column += 256) {
    const float value = __bfloat162float(input[base + column]);
    const float scale = __bfloat162float(weight[column]);
    output[base + column] = __float2bfloat16_rn(value * inverse * scale);
  }
}
