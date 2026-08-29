#include <cuda_bf16.h>
#include <stdint.h>

extern "C" __global__ void lunaflux_probe_i8_embedding_v1(
    const int32_t *, const int32_t *, const __nv_bfloat16 *, __nv_bfloat16 *) {}
extern "C" __global__ void lunaflux_probe_i8_rms_1_v1(
    const int32_t *, const __nv_bfloat16 *, const __nv_bfloat16 *,
    __nv_bfloat16 *) {}
extern "C" __global__ void lunaflux_probe_i8_qkv_v1(
    const int32_t *, const __nv_bfloat16 *, const int8_t *, const float *,
    const int8_t *, const float *, const int8_t *, const float *,
    __nv_bfloat16 *) {}
extern "C" __global__ void lunaflux_probe_i8_rope_v1(
    const int32_t *, const int32_t *, const __nv_bfloat16 *, __nv_bfloat16 *) {}
extern "C" __global__ void lunaflux_probe_i8_paged_attention_v1(
    const int32_t *, const int32_t *, const int32_t *, const int32_t *,
    const int32_t *, const int32_t *, const __nv_bfloat16 *, __nv_bfloat16 *,
    __nv_bfloat16 *, __nv_bfloat16 *) {}

extern "C" __global__ void lunaflux_probe_i8_weight_scale_projection_v1(
    const int32_t *counts,
    const __nv_bfloat16 *input,
    const int8_t *weight,
    const float *scale,
    __nv_bfloat16 *output) {
  const int token_count = counts[3];
  if (token_count <= 0 || token_count > 3) return;
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int live = (unsigned int)token_count * 4U;
  if (index >= live) return;
  const unsigned int row = index / 4U;
  const unsigned int output_channel = index % 4U;
  const unsigned int input_base = row * 4U;
  const unsigned int weight_base = output_channel * 4U;
  float accumulator = 0.0f;
  for (unsigned int inner = 0; inner < 4U; ++inner) {
    const float reconstructed = __fmul_rn((float)weight[weight_base + inner],
                                          scale[output_channel]);
    const float product = __fmul_rn(
        __bfloat162float(input[input_base + inner]), reconstructed);
    accumulator = __fadd_rn(accumulator, product);
  }
  output[index] = __float2bfloat16_rn(accumulator);
}

extern "C" __global__ void lunaflux_probe_i8_residual_1_v1(
    const int32_t *, const __nv_bfloat16 *, const __nv_bfloat16 *,
    __nv_bfloat16 *) {}
extern "C" __global__ void lunaflux_probe_i8_rms_2_v1(
    const int32_t *, const __nv_bfloat16 *, const __nv_bfloat16 *,
    __nv_bfloat16 *) {}
extern "C" __global__ void lunaflux_probe_i8_mlp_v1(
    const int32_t *, const __nv_bfloat16 *, const int8_t *, const float *,
    const int8_t *, const float *, const int8_t *, const float *,
    __nv_bfloat16 *) {}
extern "C" __global__ void lunaflux_probe_i8_residual_2_v1(
    const int32_t *, const __nv_bfloat16 *, const __nv_bfloat16 *,
    __nv_bfloat16 *) {}
extern "C" __global__ void lunaflux_probe_i8_rms_3_v1(
    const int32_t *, const __nv_bfloat16 *, const __nv_bfloat16 *,
    __nv_bfloat16 *) {}
extern "C" __global__ void lunaflux_probe_i8_lm_head_v1(
    const int32_t *, const __nv_bfloat16 *, const int8_t *, const float *,
    __nv_bfloat16 *) {}
