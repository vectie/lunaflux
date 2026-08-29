///|
fn readonly_symbol(id : @catalog.AotKernelEntryPointId) -> String {
  "lunaflux_paged_attention_bf16_readonly_v1_ep_" + id.as_int().to_string()
}

///|
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
    Some(value) => "\n  dispatch_canary[token] = " + value.to_string() + "U;"
    None => ""
  }
  @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math_constants.h>\n" +
    "static __device__ __forceinline__ float lf_readonly_cache_load_v1(\n" +
    "    const __nv_bfloat16 *cache, const int *page_indices, int page_start,\n" +
    "    int page_end, int key_position, int kv_head, int component) {\n" +
    "  const int page_slot = page_start + key_position / " +
    tokens_per_page.to_string() +
    ";\n  if (page_slot < page_start || page_slot >= page_end) return CUDART_NAN_F;\n" +
    "  const int physical_page = page_indices[page_slot];\n" +
    "  if (physical_page < 0 || (unsigned long long)physical_page >= " +
    reference.physical_page_capacity().to_string() +
    "ULL) return CUDART_NAN_F;\n" +
    "  const int token_in_page = key_position % " +
    tokens_per_page.to_string() +
    ";\n  const unsigned long long cache_index =\n" +
    "      (unsigned long long)physical_page * " +
    page_stride_values.to_string() +
    "ULL + ((unsigned long long)token_in_page * " +
    kv_heads.to_string() +
    "ULL + kv_head) * " +
    head_dimension.to_string() +
    "ULL + component;\n  return __bfloat162float(cache[cache_index]);\n}\n" +
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
    "  const int token = (int)blockIdx.x;\n" +
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
    " || token >= token_count || blockIdx.y != 0) return;\n" +
    "  int row = 0;\n" +
    "  while (row + 1 < row_count && token >= query_row_offsets[row + 1]) ++row;\n" +
    "  const int row_start = query_row_offsets[row];\n" +
    "  const int row_end = query_row_offsets[row + 1];\n" +
    "  const int page_start = page_table_offsets[row];\n" +
    "  const int page_end = page_table_offsets[row + 1];\n" +
    "  if (row_start < 0 || row_start >= row_end || row_end > token_count ||\n" +
    "      page_start < 0 || page_start >= page_end || page_end > page_entry_count) return;\n" +
    "  const int query_position = query_positions[token];\n" +
    "  if (query_position < query_positions[row_start] ||\n" +
    "      query_position >= sequence_lengths[row] ||\n" +
    "      page_end - page_start != (sequence_lengths[row] + " +
    (tokens_per_page - 1).to_string() +
    ") / " +
    tokens_per_page.to_string() +
    ") return;\n" +
    "  for (int query_head = 0; query_head < " +
    query_heads.to_string() +
    "; ++query_head) {\n" +
    "    const int kv_head = query_head / " +
    group_size.to_string() +
    ";\n    const int query_offset = token * " +
    input_row_width.to_string() +
    " + query_head * " +
    head_dimension.to_string() +
    ";\n    const float scale = rsqrtf((float)" +
    head_dimension.to_string() +
    ");\n    float maximum = -CUDART_INF_F;\n" +
    "    for (int key_position = 0; key_position <= query_position; ++key_position) {\n" +
    "      float dot = 0.0f;\n      for (int d = 0; d < " +
    head_dimension.to_string() +
    "; ++d) {\n" +
    "        dot += __bfloat162float(rotated_qkv_rows[query_offset + d]) *\n" +
    "            lf_readonly_cache_load_v1(key_cache, page_table_page_indices,\n" +
    "                page_start, page_end, key_position, kv_head, d);\n      }\n" +
    "      maximum = fmaxf(maximum, dot * scale);\n    }\n" +
    "    float denominator = 0.0f;\n" +
    "    for (int key_position = 0; key_position <= query_position; ++key_position) {\n" +
    "      float dot = 0.0f;\n      for (int d = 0; d < " +
    head_dimension.to_string() +
    "; ++d) {\n" +
    "        dot += __bfloat162float(rotated_qkv_rows[query_offset + d]) *\n" +
    "            lf_readonly_cache_load_v1(key_cache, page_table_page_indices,\n" +
    "                page_start, page_end, key_position, kv_head, d);\n      }\n" +
    "      denominator += expf(dot * scale - maximum);\n    }\n" +
    "    for (int d = 0; d < " +
    head_dimension.to_string() +
    "; ++d) {\n      float accumulator = 0.0f;\n" +
    "      for (int key_position = 0; key_position <= query_position; ++key_position) {\n" +
    "        float dot = 0.0f;\n        for (int qd = 0; qd < " +
    head_dimension.to_string() +
    "; ++qd) {\n" +
    "          dot += __bfloat162float(rotated_qkv_rows[query_offset + qd]) *\n" +
    "              lf_readonly_cache_load_v1(key_cache, page_table_page_indices,\n" +
    "                  page_start, page_end, key_position, kv_head, qd);\n        }\n" +
    "        const float value = lf_readonly_cache_load_v1(value_cache,\n" +
    "            page_table_page_indices, page_start, page_end, key_position, kv_head, d);\n" +
    "        accumulator += expf(dot * scale - maximum) * value / denominator;\n" +
    "      }\n      const int output_offset = token * " +
    query_width.to_string() +
    " + query_head * " +
    head_dimension.to_string() +
    ";\n      output[output_offset + d] = __float2bfloat16_rn(accumulator);\n" +
    "    }\n  }" +
    canary_publication +
    "\n}\n",
  )
}
