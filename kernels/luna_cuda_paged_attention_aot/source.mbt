///|
fn paged_attention_symbol(id : @catalog.AotKernelEntryPointId) -> String {
  "lunaflux_paged_attention_bf16_reference_v1_ep_" + id.as_int().to_string()
}

///|
fn paged_attention_cuda_source(
  symbol : String,
  query_heads : Int,
  kv_heads : Int,
  head_dimension : Int,
  tokens_per_page : Int,
  page_stride_bytes : Int64,
  physical_page_capacity : Int64,
  max_rows : Int,
  max_tokens : Int,
  max_page_entries : Int,
) -> Bytes {
  let query_width = query_heads * head_dimension
  let kv_width = kv_heads * head_dimension
  let qkv_width = query_width + 2 * kv_width
  let group_size = query_heads / kv_heads
  let page_stride_values = page_stride_bytes / 2L
  @utf8.encode(
    "#include <cuda_bf16.h>\n" +
    "#include <math_constants.h>\n" +
    "#define LF_BLOCK_THREADS 32\n" +
    "static __device__ __forceinline__ float lf_reduce_sum_v3(float value) {\n" +
    "  for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down_sync(0xffffffffU, value, offset);\n" +
    "  return __shfl_sync(0xffffffffU, value, 0);\n" +
    "}\n" +
    "extern \"C\" __global__ void " +
    symbol +
    "(\n" +
    "    const int *counts, const int *query_positions, const int *query_row_offsets,\n" +
    "    const int *sequence_lengths, const int *page_table_offsets,\n" +
    "    const int *page_table_page_indices, const __nv_bfloat16 *qkv,\n" +
    "    __nv_bfloat16 *output, __nv_bfloat16 *key_cache, __nv_bfloat16 *value_cache) {\n" +
    "  const int lane = (int)threadIdx.x;\n" +
    "  const int query_head = (int)blockIdx.y;\n" +
    "  if (blockDim.x != LF_BLOCK_THREADS) return;\n" +
    "  const int prefill_count = counts[0], decode_count = counts[1];\n" +
    "  const int row_count = counts[2], token_count = counts[3];\n" +
    "  const int page_entry_count = counts[4];\n" +
    "  if (prefill_count < 0 || decode_count < 0 || row_count <= 0 ||\n" +
    "      row_count != prefill_count + decode_count || row_count > " +
    max_rows.to_string() +
    " || token_count <= 0 || token_count > " +
    max_tokens.to_string() +
    " || page_entry_count <= 0 || page_entry_count > " +
    max_page_entries.to_string() +
    " || query_head >= " +
    query_heads.to_string() +
    ") return;\n" +
    "  for (int token = (int)blockIdx.x; token < token_count; token += (int)gridDim.x) {\n" +
    "  int row = 0;\n" +
    "  while (row + 1 < row_count && token >= query_row_offsets[row + 1]) ++row;\n" +
    "  const int row_start = query_row_offsets[row], row_end = query_row_offsets[row + 1];\n" +
    "  const int page_start = page_table_offsets[row], page_end = page_table_offsets[row + 1];\n" +
    "  const int output_offset = token * " +
    query_width.to_string() +
    " + query_head * " +
    head_dimension.to_string() +
    ";\n" +
    "  if (row_start < 0 || row_start >= row_end || row_end > token_count ||\n" +
    "      page_start < 0 || page_start >= page_end || page_end > page_entry_count) {\n" +
    "    if (lane == 0) output[output_offset] = __float2bfloat16_rn(CUDART_NAN_F);\n" +
    "    continue;\n" +
    "  }\n" +
    "  const int query_position = query_positions[token];\n" +
    "  const int chunk_start = query_positions[row_start];\n" +
    "  if (query_position < chunk_start || query_position >= sequence_lengths[row] ||\n" +
    "      page_end - page_start != (sequence_lengths[row] + " +
    (tokens_per_page - 1).to_string() +
    ") / " +
    tokens_per_page.to_string() +
    ") {\n" +
    "    if (lane == 0) output[output_offset] = __float2bfloat16_rn(CUDART_NAN_F);\n" +
    "    continue;\n" +
    "  }\n" +
    "  const int kv_head = query_head / " +
    group_size.to_string() +
    ";\n" +
    "  const int query_offset = token * " +
    qkv_width.to_string() +
    " + query_head * " +
    head_dimension.to_string() +
    ";\n" +
    "  const int key_qkv_offset = " +
    query_width.to_string() +
    ", value_qkv_offset = " +
    (query_width + kv_width).to_string() +
    ";\n" +
    "  if (query_head == 0) {\n" +
    "    const int page_slot = page_start + query_position / " +
    tokens_per_page.to_string() +
    ";\n" +
    "    const int physical_page = page_table_page_indices[page_slot];\n" +
    "    if (page_slot < page_end && physical_page >= 0 &&\n" +
    "        (unsigned long long)physical_page < " +
    physical_page_capacity.to_string() +
    "ULL) {\n" +
    "      const int token_in_page = query_position % " +
    tokens_per_page.to_string() +
    ";\n" +
    "      for (int linear = lane; linear < " +
    (kv_heads * head_dimension).to_string() +
    "; linear += LF_BLOCK_THREADS) {\n" +
    "        const int h = linear / " +
    head_dimension.to_string() +
    ", d = linear % " +
    head_dimension.to_string() +
    ";\n" +
    "        const unsigned long long cache_index = (unsigned long long)physical_page * " +
    page_stride_values.to_string() +
    "ULL + ((unsigned long long)token_in_page * " +
    kv_heads.to_string() +
    "ULL + h) * " +
    head_dimension.to_string() +
    "ULL + d;\n" +
    "        key_cache[cache_index] = qkv[(unsigned long long)token * " +
    qkv_width.to_string() +
    "ULL + key_qkv_offset + linear];\n" +
    "        value_cache[cache_index] = qkv[(unsigned long long)token * " +
    qkv_width.to_string() +
    "ULL + value_qkv_offset + linear];\n" +
    "      }\n" +
    "    }\n" +
    "  }\n" +
    "  const float scale = rsqrtf((float)" +
    head_dimension.to_string() +
    ");\n" +
    "  float maximum = -CUDART_INF_F, denominator = 0.0f;\n" +
    "  float accumulator[" +
    ((head_dimension + 31) / 32).to_string() +
    "];\n" +
    "  #pragma unroll\n" +
    "  for (int item = 0; item < " +
    ((head_dimension + 31) / 32).to_string() +
    "; ++item) accumulator[item] = 0.0f;\n" +
    "  for (int key_position = 0; key_position <= query_position; ++key_position) {\n" +
    "    const __nv_bfloat16 *key_vector = nullptr, *value_vector = nullptr;\n" +
    "    if (key_position >= chunk_start) {\n" +
    "      const int local = row_start + key_position - chunk_start;\n" +
    "      if (local < row_end && query_positions[local] == key_position) {\n" +
    "        const unsigned long long qkv_base = (unsigned long long)local * " +
    qkv_width.to_string() +
    "ULL + (unsigned long long)kv_head * " +
    head_dimension.to_string() +
    "ULL;\n" +
    "        key_vector = qkv + qkv_base + key_qkv_offset;\n" +
    "        value_vector = qkv + qkv_base + value_qkv_offset;\n" +
    "      }\n" +
    "    }\n" +
    "    if (key_vector == nullptr) {\n" +
    "      const int logical_page = key_position / " +
    tokens_per_page.to_string() +
    ";\n" +
    "      const int physical_page = page_table_page_indices[page_start + logical_page];\n" +
    "      if (physical_page >= 0 && (unsigned long long)physical_page < " +
    physical_page_capacity.to_string() +
    "ULL) {\n" +
    "        const int token_in_page = key_position % " +
    tokens_per_page.to_string() +
    ";\n" +
    "        const unsigned long long cache_base = (unsigned long long)physical_page * " +
    page_stride_values.to_string() +
    "ULL + ((unsigned long long)token_in_page * " +
    kv_heads.to_string() +
    "ULL + kv_head) * " +
    head_dimension.to_string() +
    "ULL;\n" +
    "        key_vector = key_cache + cache_base; value_vector = value_cache + cache_base;\n" +
    "      }\n" +
    "    }\n" +
    "    float partial = 0.0f;\n" +
    "    for (int d = lane; d < " +
    head_dimension.to_string() +
    "; d += LF_BLOCK_THREADS) {\n" +
    "      const float key = key_vector == nullptr ? CUDART_NAN_F : __bfloat162float(key_vector[d]);\n" +
    "      partial += __bfloat162float(qkv[query_offset + d]) * key;\n" +
    "    }\n" +
    "    const float score = lf_reduce_sum_v3(partial) * scale;\n" +
    "    float old_scale = 0.0f, new_scale = 0.0f;\n" +
    "    if (lane == 0) {\n" +
    "      const float next_maximum = fmaxf(maximum, score);\n" +
    "      old_scale = denominator == 0.0f ? 0.0f : expf(maximum - next_maximum);\n" +
    "      new_scale = expf(score - next_maximum);\n" +
    "      maximum = next_maximum; denominator = denominator * old_scale + new_scale;\n" +
    "    }\n" +
    "    old_scale = __shfl_sync(0xffffffffU, old_scale, 0);\n" +
    "    new_scale = __shfl_sync(0xffffffffU, new_scale, 0);\n" +
    "    #pragma unroll\n" +
    "    for (int item = 0; item < " +
    ((head_dimension + 31) / 32).to_string() +
    "; ++item) {\n" +
    "      const int d = lane + item * 32;\n" +
    "      if (d < " +
    head_dimension.to_string() +
    ") {\n" +
    "        const float value = value_vector == nullptr ? CUDART_NAN_F : __bfloat162float(value_vector[d]);\n" +
    "        accumulator[item] = accumulator[item] * old_scale + new_scale * value;\n" +
    "      }\n" +
    "    }\n" +
    "  }\n" +
    "  denominator = __shfl_sync(0xffffffffU, denominator, 0);\n" +
    "  #pragma unroll\n" +
    "  for (int item = 0; item < " +
    ((head_dimension + 31) / 32).to_string() +
    "; ++item) { const int d = lane + item * 32; if (d < " +
    head_dimension.to_string() +
    ") output[output_offset + d] = __float2bfloat16_rn(accumulator[item] / denominator); }\n" +
    "  }\n}\n",
  )
}
