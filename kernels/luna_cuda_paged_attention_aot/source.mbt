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
    "static __device__ __forceinline__ float lf_paged_load_v1(\n" +
    "    const __nv_bfloat16 *qkv, const __nv_bfloat16 *cache,\n" +
    "    const int *positions, const int *page_offsets,\n" +
    "    const int *page_indices, int row_start, int row_end,\n" +
    "    int chunk_start, int key_position, int kv_head, int component,\n" +
    "    int qkv_component_offset) {\n" +
    "  if (key_position >= chunk_start) {\n" +
    "    const int local = row_start + key_position - chunk_start;\n" +
    "    if (local < row_end && positions[local] == key_position) {\n" +
    "      const unsigned long long qkv_index =\n" +
    "          (unsigned long long)local * " +
    qkv_width.to_string() +
    "ULL + qkv_component_offset +\n" +
    "          (unsigned long long)kv_head * " +
    head_dimension.to_string() +
    "ULL + component;\n" +
    "      return __bfloat162float(qkv[qkv_index]);\n" +
    "    }\n" +
    "  }\n" +
    "  const int logical_page = key_position / " +
    tokens_per_page.to_string() +
    ";\n" +
    "  const int page_slot = page_offsets[0] + logical_page;\n" +
    "  const int physical_page = page_indices[page_slot];\n" +
    "  if (physical_page < 0 || (unsigned long long)physical_page >= " +
    physical_page_capacity.to_string() +
    "ULL) return CUDART_NAN_F;\n" +
    "  const int token_in_page = key_position % " +
    tokens_per_page.to_string() +
    ";\n" +
    "  const unsigned long long cache_index =\n" +
    "      (unsigned long long)physical_page * " +
    page_stride_values.to_string() +
    "ULL +\n" +
    "      ((unsigned long long)token_in_page * " +
    kv_heads.to_string() +
    "ULL + kv_head) * " +
    head_dimension.to_string() +
    "ULL + component;\n" +
    "  return __bfloat162float(cache[cache_index]);\n" +
    "}\n" +
    "extern \"C\" __global__ void " +
    symbol +
    "(\n" +
    "    const int *counts, const int *query_positions,\n" +
    "    const int *query_row_offsets, const int *sequence_lengths,\n" +
    "    const int *page_table_offsets, const int *page_table_page_indices,\n" +
    "    const __nv_bfloat16 *qkv, __nv_bfloat16 *output,\n" +
    "    __nv_bfloat16 *key_cache, __nv_bfloat16 *value_cache) {\n" +
    "  const int token = (int)blockIdx.x;\n" +
    "  const int query_head = (int)blockIdx.y;\n" +
    "  const int prefill_count = counts[0];\n" +
    "  const int decode_count = counts[1];\n" +
    "  const int row_count = counts[2];\n" +
    "  const int token_count = counts[3];\n" +
    "  const int page_entry_count = counts[4];\n" +
    "  if (prefill_count < 0 || decode_count < 0 ||\n" +
    "      row_count != prefill_count + decode_count || row_count <= 0 ||\n" +
    "      row_count > " +
    max_rows.to_string() +
    " || token_count <= 0 ||\n" +
    "      token_count > " +
    max_tokens.to_string() +
    " || page_entry_count <= 0 ||\n" +
    "      page_entry_count > " +
    max_page_entries.to_string() +
    " || token >= token_count ||\n" +
    "      query_head >= " +
    query_heads.to_string() +
    ") return;\n" +
    "  int row = 0;\n" +
    "  while (row + 1 < row_count && token >= query_row_offsets[row + 1]) ++row;\n" +
    "  const int row_start = query_row_offsets[row];\n" +
    "  const int row_end = query_row_offsets[row + 1];\n" +
    "  const int page_start = page_table_offsets[row];\n" +
    "  const int page_end = page_table_offsets[row + 1];\n" +
    "  const int output_offset = token * " +
    query_width.to_string() +
    " + query_head * " +
    head_dimension.to_string() +
    ";\n" +
    "  if (row_start < 0 || row_start >= row_end || row_end > token_count ||\n" +
    "      page_start < 0 || page_start >= page_end || page_end > page_entry_count) {\n" +
    "    output[output_offset] = __float2bfloat16_rn(CUDART_NAN_F);\n" +
    "    return;\n" +
    "  }\n" +
    "  const int query_position = query_positions[token];\n" +
    "  const int chunk_start = query_positions[row_start];\n" +
    "  if (query_position < chunk_start || query_position >= sequence_lengths[row] ||\n" +
    "      page_end - page_start != (sequence_lengths[row] + " +
    (tokens_per_page - 1).to_string() +
    ") / " +
    tokens_per_page.to_string() +
    ") {\n" +
    "    output[output_offset] = __float2bfloat16_rn(CUDART_NAN_F);\n" +
    "    return;\n" +
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
    ";\n" +
    "  const int value_qkv_offset = " +
    (query_width + kv_width).to_string() +
    ";\n" +
    "  if (query_head == 0) {\n" +
    "    const int logical_page = query_position / " +
    tokens_per_page.to_string() +
    ";\n" +
    "    const int page_slot = page_start + logical_page;\n" +
    "    const int physical_page = page_table_page_indices[page_slot];\n" +
    "    if (page_slot < page_end && physical_page >= 0 &&\n" +
    "        (unsigned long long)physical_page < " +
    physical_page_capacity.to_string() +
    "ULL) {\n" +
    "      const int token_in_page = query_position % " +
    tokens_per_page.to_string() +
    ";\n" +
    "      for (int h = 0; h < " +
    kv_heads.to_string() +
    "; ++h) {\n" +
    "        for (int d = 0; d < " +
    head_dimension.to_string() +
    "; ++d) {\n" +
    "          const unsigned long long cache_index =\n" +
    "              (unsigned long long)physical_page * " +
    page_stride_values.to_string() +
    "ULL +\n" +
    "              ((unsigned long long)token_in_page * " +
    kv_heads.to_string() +
    "ULL + h) * " +
    head_dimension.to_string() +
    "ULL + d;\n" +
    "          key_cache[cache_index] = qkv[(unsigned long long)token * " +
    qkv_width.to_string() +
    "ULL + key_qkv_offset + h * " +
    head_dimension.to_string() +
    " + d];\n" +
    "          value_cache[cache_index] = qkv[(unsigned long long)token * " +
    qkv_width.to_string() +
    "ULL + value_qkv_offset + h * " +
    head_dimension.to_string() +
    " + d];\n" +
    "        }\n" +
    "      }\n" +
    "    }\n" +
    "  }\n" +
    "  const float scale = rsqrtf((float)" +
    head_dimension.to_string() +
    ");\n" +
    "  float maximum = -CUDART_INF_F;\n" +
    "  for (int key_position = 0; key_position <= query_position; ++key_position) {\n" +
    "    float dot = 0.0f;\n" +
    "    for (int d = 0; d < " +
    head_dimension.to_string() +
    "; ++d) {\n" +
    "      const float key = lf_paged_load_v1(qkv, key_cache, query_positions,\n" +
    "          page_table_offsets + row, page_table_page_indices, row_start, row_end,\n" +
    "          chunk_start, key_position, kv_head, d, key_qkv_offset);\n" +
    "      dot += __bfloat162float(qkv[query_offset + d]) * key;\n" +
    "    }\n" +
    "    maximum = fmaxf(maximum, dot * scale);\n" +
    "  }\n" +
    "  float denominator = 0.0f;\n" +
    "  for (int key_position = 0; key_position <= query_position; ++key_position) {\n" +
    "    float dot = 0.0f;\n" +
    "    for (int d = 0; d < " +
    head_dimension.to_string() +
    "; ++d) {\n" +
    "      const float key = lf_paged_load_v1(qkv, key_cache, query_positions,\n" +
    "          page_table_offsets + row, page_table_page_indices, row_start, row_end,\n" +
    "          chunk_start, key_position, kv_head, d, key_qkv_offset);\n" +
    "      dot += __bfloat162float(qkv[query_offset + d]) * key;\n" +
    "    }\n" +
    "    denominator += expf(dot * scale - maximum);\n" +
    "  }\n" +
    "  for (int d = 0; d < " +
    head_dimension.to_string() +
    "; ++d) {\n" +
    "    float accumulator = 0.0f;\n" +
    "    for (int key_position = 0; key_position <= query_position; ++key_position) {\n" +
    "      float dot = 0.0f;\n" +
    "      for (int qd = 0; qd < " +
    head_dimension.to_string() +
    "; ++qd) {\n" +
    "        const float key = lf_paged_load_v1(qkv, key_cache, query_positions,\n" +
    "            page_table_offsets + row, page_table_page_indices, row_start, row_end,\n" +
    "            chunk_start, key_position, kv_head, qd, key_qkv_offset);\n" +
    "        dot += __bfloat162float(qkv[query_offset + qd]) * key;\n" +
    "      }\n" +
    "      const float value = lf_paged_load_v1(qkv, value_cache, query_positions,\n" +
    "          page_table_offsets + row, page_table_page_indices, row_start, row_end,\n" +
    "          chunk_start, key_position, kv_head, d, value_qkv_offset);\n" +
    "      accumulator += expf(dot * scale - maximum) * value / denominator;\n" +
    "    }\n" +
    "    output[output_offset + d] = __float2bfloat16_rn(accumulator);\n" +
    "  }\n" +
    "}\n",
  )
}
