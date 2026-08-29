#include <cuda_bf16.h>
#include <math_constants.h>
static __device__ __forceinline__ float lf_paged_load_v1(
    const __nv_bfloat16 *qkv, const __nv_bfloat16 *cache,
    const int *positions, const int *page_offsets,
    const int *page_indices, int row_start, int row_end,
    int chunk_start, int key_position, int kv_head, int component,
    int qkv_component_offset) {
  if (key_position >= chunk_start) {
    const int local = row_start + key_position - chunk_start;
    if (local < row_end && positions[local] == key_position) {
      const unsigned long long qkv_index =
          (unsigned long long)local * 8ULL + qkv_component_offset +
          (unsigned long long)kv_head * 2ULL + component;
      return __bfloat162float(qkv[qkv_index]);
    }
  }
  const int logical_page = key_position / 2;
  const int page_slot = page_offsets[0] + logical_page;
  const int physical_page = page_indices[page_slot];
  if (physical_page < 0 || (unsigned long long)physical_page >= 4ULL) return CUDART_NAN_F;
  const int token_in_page = key_position % 2;
  const unsigned long long cache_index =
      (unsigned long long)physical_page * 128ULL +
      ((unsigned long long)token_in_page * 1ULL + kv_head) * 2ULL + component;
  return __bfloat162float(cache[cache_index]);
}
extern "C" __global__ void lunaflux_paged_attention_bf16_reference_v1_ep_3(
    const int *counts, const int *query_positions,
    const int *query_row_offsets, const int *sequence_lengths,
    const int *page_table_offsets, const int *page_table_page_indices,
    const __nv_bfloat16 *qkv, __nv_bfloat16 *output,
    __nv_bfloat16 *key_cache, __nv_bfloat16 *value_cache) {
  const int token = (int)blockIdx.x;
  const int query_head = (int)blockIdx.y;
  const int prefill_count = counts[0];
  const int decode_count = counts[1];
  const int row_count = counts[2];
  const int token_count = counts[3];
  const int page_entry_count = counts[4];
  if (prefill_count < 0 || decode_count < 0 ||
      row_count != prefill_count + decode_count || row_count <= 0 ||
      row_count > 2 || token_count <= 0 ||
      token_count > 3 || page_entry_count <= 0 ||
      page_entry_count > 3 || token >= token_count ||
      query_head >= 2) return;
  int row = 0;
  while (row + 1 < row_count && token >= query_row_offsets[row + 1]) ++row;
  const int row_start = query_row_offsets[row];
  const int row_end = query_row_offsets[row + 1];
  const int page_start = page_table_offsets[row];
  const int page_end = page_table_offsets[row + 1];
  const int output_offset = token * 4 + query_head * 2;
  if (row_start < 0 || row_start >= row_end || row_end > token_count ||
      page_start < 0 || page_start >= page_end || page_end > page_entry_count) {
    output[output_offset] = __float2bfloat16_rn(CUDART_NAN_F);
    return;
  }
  const int query_position = query_positions[token];
  const int chunk_start = query_positions[row_start];
  if (query_position < chunk_start || query_position >= sequence_lengths[row] ||
      page_end - page_start != (sequence_lengths[row] + 1) / 2) {
    output[output_offset] = __float2bfloat16_rn(CUDART_NAN_F);
    return;
  }
  const int kv_head = query_head / 2;
  const int query_offset = token * 8 + query_head * 2;
  const int key_qkv_offset = 4;
  const int value_qkv_offset = 6;
  if (query_head == 0) {
    const int logical_page = query_position / 2;
    const int page_slot = page_start + logical_page;
    const int physical_page = page_table_page_indices[page_slot];
    if (page_slot < page_end && physical_page >= 0 &&
        (unsigned long long)physical_page < 4ULL) {
      const int token_in_page = query_position % 2;
      for (int h = 0; h < 1; ++h) {
        for (int d = 0; d < 2; ++d) {
          const unsigned long long cache_index =
              (unsigned long long)physical_page * 128ULL +
              ((unsigned long long)token_in_page * 1ULL + h) * 2ULL + d;
          key_cache[cache_index] = qkv[(unsigned long long)token * 8ULL + key_qkv_offset + h * 2 + d];
          value_cache[cache_index] = qkv[(unsigned long long)token * 8ULL + value_qkv_offset + h * 2 + d];
        }
      }
    }
  }
  const float scale = rsqrtf((float)2);
  float maximum = -CUDART_INF_F;
  for (int key_position = 0; key_position <= query_position; ++key_position) {
    float dot = 0.0f;
    for (int d = 0; d < 2; ++d) {
      const float key = lf_paged_load_v1(qkv, key_cache, query_positions,
          page_table_offsets + row, page_table_page_indices, row_start, row_end,
          chunk_start, key_position, kv_head, d, key_qkv_offset);
      dot += __bfloat162float(qkv[query_offset + d]) * key;
    }
    maximum = fmaxf(maximum, dot * scale);
  }
  float denominator = 0.0f;
  for (int key_position = 0; key_position <= query_position; ++key_position) {
    float dot = 0.0f;
    for (int d = 0; d < 2; ++d) {
      const float key = lf_paged_load_v1(qkv, key_cache, query_positions,
          page_table_offsets + row, page_table_page_indices, row_start, row_end,
          chunk_start, key_position, kv_head, d, key_qkv_offset);
      dot += __bfloat162float(qkv[query_offset + d]) * key;
    }
    denominator += expf(dot * scale - maximum);
  }
  for (int d = 0; d < 2; ++d) {
    float accumulator = 0.0f;
    for (int key_position = 0; key_position <= query_position; ++key_position) {
      float dot = 0.0f;
      for (int qd = 0; qd < 2; ++qd) {
        const float key = lf_paged_load_v1(qkv, key_cache, query_positions,
            page_table_offsets + row, page_table_page_indices, row_start, row_end,
            chunk_start, key_position, kv_head, qd, key_qkv_offset);
        dot += __bfloat162float(qkv[query_offset + qd]) * key;
      }
      const float value = lf_paged_load_v1(qkv, value_cache, query_positions,
          page_table_offsets + row, page_table_page_indices, row_start, row_end,
          chunk_start, key_position, kv_head, d, value_qkv_offset);
      accumulator += expf(dot * scale - maximum) * value / denominator;
    }
    output[output_offset + d] = __float2bfloat16_rn(accumulator);
  }
}
