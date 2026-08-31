#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
extern "C" __global__ void lunaflux_luna_gated_mlp_bf16_reference_v1_op_5_ep_6(const int32_t *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *gate_weight, const __nv_bfloat16 *up_weight, const __nv_bfloat16 *down_weight, __nv_bfloat16 *output) {
  const int token_count = counts[3];
  if (token_count <= 0 || token_count > 4) return;
  const int token = (int)blockIdx.x;
  if (token >= token_count) return;
  extern __shared__ float combined[];
  const unsigned long long input_base = (unsigned long long)token * 4ULL;
  for (int intermediate = (int)threadIdx.x; intermediate < 8; intermediate += (int)blockDim.x) {
    const unsigned long long projection_base = (unsigned long long)intermediate * 4ULL;
    float gate = 0.0f;
    float up = 0.0f;
    for (int inner = 0; inner < 4; ++inner) {
      const float input_value = __bfloat162float(input[input_base + inner]);
      gate = __fadd_rn(gate, __fmul_rn(input_value, __bfloat162float(gate_weight[projection_base + inner])));
      up = __fadd_rn(up, __fmul_rn(input_value, __bfloat162float(up_weight[projection_base + inner])));
    }
    const float silu = gate / __fadd_rn(1.0f, expf(-gate));
    combined[intermediate] = __fmul_rn(silu, up);
  }
  __syncthreads();
  for (int output_column = (int)threadIdx.x; output_column < 4; output_column += (int)blockDim.x) {
    const unsigned long long down_base = (unsigned long long)output_column * 8ULL;
    float output_accumulator = 0.0f;
    for (int intermediate = 0; intermediate < 8; ++intermediate) {
      const float down = __bfloat162float(down_weight[down_base + intermediate]);
      output_accumulator = __fadd_rn(output_accumulator, __fmul_rn(combined[intermediate], down));
    }
    output[(unsigned long long)token * 4ULL + output_column] = __float2bfloat16_rn(output_accumulator);
  }
}
