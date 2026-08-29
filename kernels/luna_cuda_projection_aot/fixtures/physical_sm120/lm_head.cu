#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
extern "C" __global__ void lunaflux_luna_language_model_head_bf16_reference_v1_op_6_ep_7(const int32_t *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *weight, __nv_bfloat16 *output) {
  const int token_count = counts[3];
  if (token_count <= 0 || token_count > 4) return;
  const unsigned long long index = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;
  const unsigned long long live = (unsigned long long)token_count * 16ULL;
  if (index >= live) return;
  const unsigned long long row = index / 16ULL;
  const unsigned long long column = index % 16ULL;
  const unsigned long long input_base = row * 4ULL;
  const unsigned long long weight_base = column * 4ULL;
  float accumulator = 0.0f;
  for (int inner = 0; inner < 4; ++inner) {
    const float product = __fmul_rn(__bfloat162float(input[input_base + inner]), __bfloat162float(weight[weight_base + inner]));
    accumulator = __fadd_rn(accumulator, product);
  }
  output[index] = __float2bfloat16_rn(accumulator);
}
