#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
extern "C" __global__ void lunaflux_luna_gated_mlp_bf16_reference_v1_op_5_ep_6(const int32_t *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *gate_weight, const __nv_bfloat16 *up_weight, const __nv_bfloat16 *down_weight, __nv_bfloat16 *output) {
  const int token_count = counts[3];
  if (token_count <= 0 || token_count > 4) return;
  const unsigned long long index = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;
  const unsigned long long live = (unsigned long long)token_count * 4ULL;
  if (index >= live) return;
  const unsigned long long row = index / 4ULL;
  const unsigned long long output_column = index % 4ULL;
  const unsigned long long input_base = row * 4ULL;
  const unsigned long long down_base = output_column * 8ULL;
  float output_accumulator = 0.0f;
  for (int intermediate = 0; intermediate < 8; ++intermediate) {
    const unsigned long long projection_base = (unsigned long long)intermediate * 4ULL;
    float gate = 0.0f;
    float up = 0.0f;
    for (int inner = 0; inner < 4; ++inner) {
      const float input_value = __bfloat162float(input[input_base + inner]);
      gate = __fadd_rn(gate, __fmul_rn(input_value, __bfloat162float(gate_weight[projection_base + inner])));
      up = __fadd_rn(up, __fmul_rn(input_value, __bfloat162float(up_weight[projection_base + inner])));
    }
    const float silu = gate / __fadd_rn(1.0f, expf(-gate));
    const float combined = __fmul_rn(silu, up);
    const float down = __bfloat162float(down_weight[down_base + intermediate]);
    output_accumulator = __fadd_rn(output_accumulator, __fmul_rn(combined, down));
  }
  output[index] = __float2bfloat16_rn(output_accumulator);
}
