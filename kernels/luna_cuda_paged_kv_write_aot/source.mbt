///|
fn paged_kv_write_source(
  symbol : String,
  query_heads : Int,
  kv_heads : Int,
  head_dimension : Int,
  kv_width : Int,
  tokens_per_page : Int,
  total_page_count : Int,
  page_stride_bytes : Int64,
  max_rows : Int,
  max_tokens : Int,
  max_page_entries : Int,
  dispatch_canary : UInt,
) -> Bytes {
  let qkv_width = (query_heads + 2 * kv_heads) * head_dimension
  let key_begin = query_heads * head_dimension
  let value_begin = (query_heads + kv_heads) * head_dimension
  @utf8.encode(
    "#include <cuda_bf16.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    symbol +
    "(\n" +
    "  const int32_t* step_counts, const int32_t* query_positions,\n" +
    "  const int32_t* query_row_offsets, const int32_t* sequence_lengths,\n" +
    "  const int32_t* page_table_offsets, const int32_t* page_table_page_indices,\n" +
    "  const __nv_bfloat16* positioned_qkv,\n" +
    "  __nv_bfloat16* key_cache, __nv_bfloat16* value_cache,\n" +
    "  unsigned int* dispatch_canary) {\n" +
    "  if (blockDim.x != 256 || blockIdx.y != 0 || blockIdx.z != 0) return;\n" +
    "  const int token = (int)blockIdx.x;\n" +
    "  const int component = (int)threadIdx.x;\n" +
    "  const int prefill = step_counts[0];\n" +
    "  const int decode = step_counts[1];\n" +
    "  const int rows = step_counts[2];\n" +
    "  const int tokens = step_counts[3];\n" +
    "  const int table_cells = step_counts[4];\n" +
    "  if (prefill < 0 || decode < 0 ||\n" +
    "      (long long)rows != (long long)prefill + (long long)decode ||\n" +
    "      rows <= 0 || rows > " +
    max_rows.to_string() +
    " || tokens <= 0 || tokens > " +
    max_tokens.to_string() +
    " ||\n" +
    "      table_cells <= 0 || table_cells > " +
    max_page_entries.to_string() +
    " || token >= tokens) return;\n" +
    "  if (query_row_offsets[0] != 0 || query_row_offsets[rows] != tokens ||\n" +
    "      page_table_offsets[0] != 0 || page_table_offsets[rows] != table_cells) return;\n" +
    "  for (int r = 0; r < rows; ++r) {\n" +
    "    if (query_row_offsets[r] < 0 || query_row_offsets[r] >= query_row_offsets[r + 1] ||\n" +
    "        query_row_offsets[r + 1] > tokens || sequence_lengths[r] <= 0 ||\n" +
    "        page_table_offsets[r] < 0 || page_table_offsets[r] >= page_table_offsets[r + 1] ||\n" +
    "        page_table_offsets[r + 1] > table_cells) return;\n" +
    "  }\n" +
    "  int row = 0;\n" +
    "  while (row + 1 < rows && token >= query_row_offsets[row + 1]) ++row;\n" +
    "  const int row_begin = query_row_offsets[row];\n" +
    "  const int row_end = query_row_offsets[row + 1];\n" +
    "  if (token < row_begin || token >= row_end) return;\n" +
    "  const int position = query_positions[token];\n" +
    "  if (position < 0 || position >= sequence_lengths[row]) return;\n" +
    "  const int logical_page = position / " +
    tokens_per_page.to_string() +
    ";\n" +
    "  const int begin = page_table_offsets[row];\n" +
    "  const int end = page_table_offsets[row + 1];\n" +
    "  if (logical_page < 0 || logical_page >= end - begin) return;\n" +
    "  const int cell = begin + logical_page;\n" +
    "  if (cell >= table_cells) return;\n" +
    "  const int page = page_table_page_indices[cell];\n" +
    "  if (page < 0 || page >= " +
    total_page_count.to_string() +
    ") return;\n" +
    "  const unsigned long long cache_index = (unsigned long long)page * " +
    (page_stride_bytes / 2L).to_string() +
    "ULL +\n" +
    "    (unsigned long long)(position % " +
    tokens_per_page.to_string() +
    ") * " +
    kv_width.to_string() +
    "ULL + (unsigned long long)component;\n" +
    "  const unsigned long long qkv_base = (unsigned long long)token * " +
    qkv_width.to_string() +
    "ULL;\n" +
    "  if (component < " +
    kv_width.to_string() +
    ") {\n" +
    "    key_cache[cache_index] = positioned_qkv[qkv_base + " +
    key_begin.to_string() +
    "ULL + component];\n" +
    "    value_cache[cache_index] = positioned_qkv[qkv_base + " +
    value_begin.to_string() +
    "ULL + component];\n" +
    "  }\n  __syncthreads();\n" +
    "  if (component == 0) atomicAdd(dispatch_canary, " +
    dispatch_canary.to_string() +
    "U);\n}\n",
  )
}
