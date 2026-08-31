///|
fn readonly_symbol(id : @catalog.AotKernelEntryPointId) -> String {
  "lunaflux_paged_attention_bf16_readonly_v1_ep_" + id.as_int().to_string()
}

///|
/// Emits one warp per active query-head tile. The preceding fused ingress
/// kernel has already published every current-token K/V row into the paged
/// cache, so this kernel only traverses cache pages and performs online
/// softmax; no current-chunk fallback or KV mutation remains.
fn readonly_cuda_source(
  symbol : String,
  reference : @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering,
  input_row_width : Int,
  diagnostic_canary_per_token : Int?,
) -> Bytes {
  let query_heads = reference.query_heads()
  let kv_heads = reference.kv_heads()
  let head_dimension = reference.head_dimension()
  let tokens_per_page = reference.tokens_per_page()
  let query_width = query_heads * head_dimension
  let group_size = query_heads / kv_heads
  let page_stride_values = reference.page_stride_bytes() / 2L
  let profile = reference.profile()
  let canary_argument = match diagnostic_canary_per_token {
    Some(_) => ",\n    unsigned int *dispatch_canary"
    None => ""
  }
  let canary_publication = match diagnostic_canary_per_token {
    Some(value) =>
      "\n    if (query_head == 0 && lane == 0) dispatch_canary[token] = " +
      value.to_string() +
      "U;"
    None => ""
  }
  @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math_constants.h>\n" +
    "#define LF_READONLY_WARP_THREADS 32\n" +
    "static __device__ __forceinline__ float lf_readonly_reduce_sum_v3(float value) {\n" +
    "  for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down_sync(0xffffffffU, value, offset);\n" +
    "  return __shfl_sync(0xffffffffU, value, 0);\n}\n" +
    "extern \"C\" __global__ void " +
    symbol +
    "(\n" +
    "    const int *counts, const int *query_positions,\n" +
    "    const int *query_row_offsets, const int *sequence_lengths,\n" +
    "    const int *page_table_offsets, const int *page_table_page_indices,\n" +
    "    const __nv_bfloat16 *rotated_qkv_rows, __nv_bfloat16 *output,\n" +
    "    const __nv_bfloat16 *key_cache, const __nv_bfloat16 *value_cache" +
    canary_argument +
    ") {\n" +
    "  const int lane = (int)threadIdx.x, query_head = (int)blockIdx.y;\n" +
    "  if (blockDim.x != LF_READONLY_WARP_THREADS || query_head >= " +
    query_heads.to_string() +
    ") return;\n" +
    "  const int prefill_count = counts[0], decode_count = counts[1];\n" +
    "  const int row_count = counts[2], token_count = counts[3];\n" +
    "  const int page_entry_count = counts[4];\n" +
    "  if (prefill_count < 0 || decode_count < 0 ||\n" +
    "      row_count != prefill_count + decode_count || row_count <= 0 ||\n" +
    "      row_count > " +
    profile.max_query_rows().to_string() +
    " || token_count <= 0 || token_count > " +
    profile.max_query_tokens().to_string() +
    " || page_entry_count <= 0 || page_entry_count > " +
    profile.max_page_table_entries().to_string() +
    ") return;\n" +
    "  for (int token = (int)blockIdx.x; token < token_count; token += (int)gridDim.x) {\n" +
    "    int row = 0;\n" +
    "    while (row + 1 < row_count && token >= query_row_offsets[row + 1]) ++row;\n" +
    "    const int row_start = query_row_offsets[row], row_end = query_row_offsets[row + 1];\n" +
    "    const int page_start = page_table_offsets[row], page_end = page_table_offsets[row + 1];\n" +
    "    const int output_offset = token * " +
    query_width.to_string() +
    " + query_head * " +
    head_dimension.to_string() +
    ";\n" +
    "    if (row_start < 0 || row_start >= row_end || row_end > token_count ||\n" +
    "        page_start < 0 || page_start >= page_end || page_end > page_entry_count) {\n" +
    "      for (int d = lane; d < " +
    head_dimension.to_string() +
    "; d += LF_READONLY_WARP_THREADS) output[output_offset + d] = __float2bfloat16_rn(CUDART_NAN_F);\n" +
    "      continue;\n    }\n" +
    "    const int query_position = query_positions[token];\n" +
    "    if (query_position < query_positions[row_start] ||\n" +
    "        query_position >= sequence_lengths[row] ||\n" +
    "        page_end - page_start != (sequence_lengths[row] + " +
    (tokens_per_page - 1).to_string() +
    ") / " +
    tokens_per_page.to_string() +
    ") {\n" +
    "      for (int d = lane; d < " +
    head_dimension.to_string() +
    "; d += LF_READONLY_WARP_THREADS) output[output_offset + d] = __float2bfloat16_rn(CUDART_NAN_F);\n" +
    "      continue;\n    }\n" +
    "    const int kv_head = query_head / " +
    group_size.to_string() +
    ";\n" +
    "    const int query_offset = token * " +
    input_row_width.to_string() +
    " + query_head * " +
    head_dimension.to_string() +
    ";\n" +
    "    const float scale = rsqrtf((float)" +
    head_dimension.to_string() +
    ");\n" +
    "    float maximum = -CUDART_INF_F, denominator = 0.0f;\n" +
    "    float accumulator[" +
    ((head_dimension + 31) / 32).to_string() +
    "];\n" +
    "    #pragma unroll\n" +
    "    for (int item = 0; item < " +
    ((head_dimension + 31) / 32).to_string() +
    "; ++item) accumulator[item] = 0.0f;\n" +
    "    bool valid = true;\n" +
    "    for (int key_position = 0; key_position <= query_position; ++key_position) {\n" +
    "      const int logical_page = key_position / " +
    tokens_per_page.to_string() +
    ";\n" +
    "      const int physical_page = page_table_page_indices[page_start + logical_page];\n" +
    "      if (physical_page < 0 || (unsigned long long)physical_page >= " +
    reference.physical_page_capacity().to_string() +
    "ULL) { valid = false; break; }\n" +
    "      const int token_in_page = key_position % " +
    tokens_per_page.to_string() +
    ";\n" +
    "      const unsigned long long cache_base = (unsigned long long)physical_page * " +
    page_stride_values.to_string() +
    "ULL + ((unsigned long long)token_in_page * " +
    kv_heads.to_string() +
    "ULL + kv_head) * " +
    head_dimension.to_string() +
    "ULL;\n" +
    "      const __nv_bfloat16 *key_vector = key_cache + cache_base;\n" +
    "      const __nv_bfloat16 *value_vector = value_cache + cache_base;\n" +
    "      float partial = 0.0f;\n" +
    "      for (int d = lane; d < " +
    head_dimension.to_string() +
    "; d += LF_READONLY_WARP_THREADS)\n" +
    "        partial += __bfloat162float(rotated_qkv_rows[query_offset + d]) * __bfloat162float(key_vector[d]);\n" +
    "      const float score = lf_readonly_reduce_sum_v3(partial) * scale;\n" +
    "      float old_scale = 0.0f, new_scale = 0.0f;\n" +
    "      if (lane == 0) {\n" +
    "        const float next_maximum = fmaxf(maximum, score);\n" +
    "        old_scale = denominator == 0.0f ? 0.0f : expf(maximum - next_maximum);\n" +
    "        new_scale = expf(score - next_maximum);\n" +
    "        maximum = next_maximum; denominator = denominator * old_scale + new_scale;\n" +
    "      }\n" +
    "      old_scale = __shfl_sync(0xffffffffU, old_scale, 0);\n" +
    "      new_scale = __shfl_sync(0xffffffffU, new_scale, 0);\n" +
    "      #pragma unroll\n" +
    "      for (int item = 0; item < " +
    ((head_dimension + 31) / 32).to_string() +
    "; ++item) {\n" +
    "        const int d = lane + item * LF_READONLY_WARP_THREADS;\n" +
    "        if (d < " +
    head_dimension.to_string() +
    ") accumulator[item] = accumulator[item] * old_scale + new_scale * __bfloat162float(value_vector[d]);\n" +
    "      }\n    }\n" +
    "    denominator = __shfl_sync(0xffffffffU, denominator, 0);\n" +
    "    if (!valid || denominator <= 0.0f) {\n" +
    "      for (int d = lane; d < " +
    head_dimension.to_string() +
    "; d += LF_READONLY_WARP_THREADS) output[output_offset + d] = __float2bfloat16_rn(CUDART_NAN_F);\n" +
    "      continue;\n    }\n" +
    "    #pragma unroll\n" +
    "    for (int item = 0; item < " +
    ((head_dimension + 31) / 32).to_string() +
    "; ++item) {\n" +
    "      const int d = lane + item * LF_READONLY_WARP_THREADS;\n" +
    "      if (d < " +
    head_dimension.to_string() +
    ") output[output_offset + d] = __float2bfloat16_rn(accumulator[item] / denominator);\n" +
    "    }" +
    canary_publication +
    "\n  }\n}\n",
  )
}
