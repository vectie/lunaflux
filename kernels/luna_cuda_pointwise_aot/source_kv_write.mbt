///|
fn kv_write_source(symbol : String, parameters : PointwiseParameters) -> Bytes {
  let query_width = parameters.query_heads * parameters.head_dimension
  let kv_width = parameters.kv_heads * parameters.head_dimension
  let page_stride_elements = parameters.page_stride_bytes / 2L
  @utf8.encode(
    cuda_prelude() +
    "extern \"C\" __global__ void " +
    symbol +
    "(const int32_t *counts, const int32_t *positions, " +
    "const int32_t *row_offsets, const int32_t *sequence_lengths, " +
    "const int32_t *page_offsets, const int32_t *physical_pages, " +
    "const __nv_bfloat16 *rotated_qkv, __nv_bfloat16 *key_cache, " +
    "__nv_bfloat16 *value_cache) {\n" +
    "  (void)sequence_lengths;\n" +
    "  const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;\n" +
    "  const int live_tokens = counts[3];\n  if (live_tokens <= 0 || counts[2] <= 0 || counts[4] <= 0) return;\n" +
    "  const unsigned long long live = (unsigned long long)live_tokens * " +
    kv_width.to_string() +
    "ULL;\n  if (i >= live) return;\n" +
    "  const int token = (int)(i / " +
    kv_width.to_string() +
    "ULL);\n" +
    "  const int component = (int)(i % " +
    kv_width.to_string() +
    "ULL);\n" +
    "  int row = 0;\n  while (row < counts[2] && token >= row_offsets[row + 1]) ++row;\n" +
    "  if (row >= counts[2]) return;\n" +
    "  const int position = positions[token];\n  if (position < 0) return;\n" +
    "  const int logical_page = position / " +
    parameters.tokens_per_page.to_string() +
    ";\n" +
    "  const int table_index = page_offsets[row] + logical_page;\n" +
    "  if (table_index < page_offsets[row] || table_index >= page_offsets[row + 1] || table_index >= counts[4]) return;\n" +
    "  const int physical_page = physical_pages[table_index];\n" +
    "  if (physical_page < 0 || physical_page >= " +
    parameters.total_page_count.to_string() +
    ") return;\n" +
    "  const unsigned long long cache_index = (unsigned long long)physical_page * " +
    page_stride_elements.to_string() +
    "ULL + " +
    "(unsigned long long)(position % " +
    parameters.tokens_per_page.to_string() +
    ") * " +
    kv_width.to_string() +
    "ULL + component;\n" +
    "  const unsigned long long input_base = (unsigned long long)token * " +
    parameters.width.to_string() +
    "ULL;\n" +
    "  key_cache[cache_index] = rotated_qkv[input_base + " +
    query_width.to_string() +
    "ULL + component];\n" +
    "  value_cache[cache_index] = rotated_qkv[input_base + " +
    (query_width + kv_width).to_string() +
    "ULL + component];\n}\n",
  )
}
