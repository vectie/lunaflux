#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
extern "C" __global__ void lunaflux_luna_qkv_projection_bf16_reference_v1_op_1_ep_2(const int32_t *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *query_weight, const __nv_bfloat16 *key_weight, const __nv_bfloat16 *value_weight, __nv_bfloat16 *output) {
  const int token_count = counts[3];
  if (token_count <= 0 || token_count > 4) return;
  const unsigned long long index = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;
  const unsigned long long live = (unsigned long long)token_count * 8ULL;
  if (index >= live) return;
  const unsigned long long row = index / 8ULL;
  const int column = (int)(index % 8ULL);
  const __nv_bfloat16 *weight = query_weight;
  int local_column = column;
  if (column >= 4) {
    local_column = column - 4;
    if (local_column >= 2) { weight = value_weight; local_column -= 2; }
    else { weight = key_weight; }
  }
  const unsigned long long input_base = row * 4ULL;
  const unsigned long long weight_base = (unsigned long long)local_column * 4ULL;
  float accumulator = 0.0f;
  for (int inner = 0; inner < 4; ++inner) {
    const float product = __fmul_rn(__bfloat162float(input[input_base + inner]), __bfloat162float(weight[weight_base + inner]));
    accumulator = __fadd_rn(accumulator, product);
  }
  output[index] = __float2bfloat16_rn(accumulator);
}
