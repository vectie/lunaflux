#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

extern "C" __global__ void lunaflux_probe_embedding_h4(
    const int32_t *counts, const int32_t *token_ids,
    const __nv_bfloat16 *table, __nv_bfloat16 *output) {
  const unsigned long long index =
      (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;
  const int live_tokens = counts[3];
  if (live_tokens <= 0 || live_tokens > 3) return;
  const unsigned long long live = (unsigned long long)live_tokens * 4ULL;
  if (index < live) {
    const int token = token_ids[index / 4ULL];
    output[index] = token >= 0 && token < 16
        ? table[(unsigned long long)token * 4ULL + index % 4ULL]
        : __float2bfloat16_rn(0.0f);
  }
}

extern "C" __global__ void lunaflux_probe_rms_norm_h4(
    const int32_t *counts, const __nv_bfloat16 *input,
    const __nv_bfloat16 *weight, __nv_bfloat16 *output) {
  const int row = (int)blockIdx.x;
  if (counts[3] <= 0 || counts[3] > 3 || row >= counts[3]) return;
  __shared__ float partial[256];
  float sum = 0.0f;
  const unsigned long long base = (unsigned long long)row * 4ULL;
  for (int column = threadIdx.x; column < 4; column += 256) {
    const float value = __bfloat162float(input[base + column]);
    sum = __fadd_rn(sum, __fmul_rn(value, value));
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] =
          __fadd_rn(partial[threadIdx.x], partial[threadIdx.x + stride]);
    __syncthreads();
  }
  const float inverse = rsqrtf(partial[0] / 4.0f + 0.00001f);
  for (int column = threadIdx.x; column < 4; column += 256) {
    const float value = __bfloat162float(input[base + column]);
    const float scale = __bfloat162float(weight[column]);
    output[base + column] = __float2bfloat16_rn(
        __fmul_rn(__fmul_rn(value, inverse), scale));
  }
}

extern "C" __global__ void lunaflux_probe_rope_qkv_h4(
    const int32_t *counts, const int32_t *positions,
    const __nv_bfloat16 *input, __nv_bfloat16 *output) {
  const unsigned long long index =
      (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;
  const int live_tokens = counts[3];
  if (live_tokens <= 0 || live_tokens > 3 ||
      index >= (unsigned long long)live_tokens * 8ULL) return;
  const int column = (int)(index % 8ULL);
  if (column >= 6) {
    output[index] = input[index];
    return;
  }
  const int pair_base = column & ~1;
  const unsigned long long row_base = index - (unsigned long long)column;
  const float angle = (float)positions[index / 8ULL];
  const float cosine = cosf(angle);
  const float sine = sinf(angle);
  const float first = __bfloat162float(input[row_base + pair_base]);
  const float second = __bfloat162float(input[row_base + pair_base + 1]);
  const float value = (column & 1) == 0
      ? __fadd_rn(__fmul_rn(first, cosine), -__fmul_rn(second, sine))
      : __fadd_rn(__fmul_rn(second, cosine), __fmul_rn(first, sine));
  output[index] = __float2bfloat16_rn(value);
}

extern "C" __global__ void lunaflux_probe_residual_h4(
    const int32_t *counts, const __nv_bfloat16 *left,
    const __nv_bfloat16 *right, __nv_bfloat16 *output) {
  const unsigned long long index =
      (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;
  const int live_tokens = counts[3];
  if (live_tokens <= 0 || live_tokens > 3 ||
      index >= (unsigned long long)live_tokens * 4ULL) return;
  output[index] = __float2bfloat16_rn(__fadd_rn(
      __bfloat162float(left[index]), __bfloat162float(right[index])));
}
