#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
extern "C" __global__ void lunaflux_luna_embedding_lookup_v1_op_0_ep_1(const int32_t *counts, const int32_t *token_ids, const __nv_bfloat16 *table, __nv_bfloat16 *output) {
  const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;
  const int live_tokens = counts[3];
  if (live_tokens <= 0) return;
  const unsigned long long live = (unsigned long long)live_tokens * 8ULL;
  if (i < live) {
    const int32_t token = token_ids[i / 8ULL];
    if (token >= 0 && token < 16) {
      output[i] = table[(unsigned long long)token * 8ULL + i % 8ULL];
    } else { output[i] = __float2bfloat16_rn(0.0f); }
  }
}
