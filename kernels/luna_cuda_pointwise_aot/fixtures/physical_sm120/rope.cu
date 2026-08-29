#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
extern "C" __global__ void lunaflux_luna_positioned_rotary_v1_op_3_ep_4(const int32_t *counts, const int32_t *positions, const __nv_bfloat16 *input, __nv_bfloat16 *output) {
  const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;
  const int live_tokens = counts[3];
  if (live_tokens <= 0) return;
  const unsigned long long live = (unsigned long long)live_tokens * 16ULL;
  if (i >= live) return;
  const int column = (int)(i % 16ULL);
  if (column >= 12) { output[i] = input[i]; return; }
  const int lane = column % 4;
  const int pair = lane % 2;
  const int head_base = column - lane;
  const unsigned long long row_base = i - (unsigned long long)column;
  const float exponent = (float)(pair * 2) / 4.0f;
  const float angle = (float)positions[i / 16ULL] / powf(10000.0f, exponent);
  const float cosine = cosf(angle);
  const float sine = sinf(angle);
  const float first = __bfloat162float(input[row_base + head_base + pair]);
  const float second = __bfloat162float(input[row_base + head_base + pair + 2]);
  const float value = lane < 2 ? first * cosine - second * sine : second * cosine + first * sine;
  output[i] = __float2bfloat16_rn(value);
}
