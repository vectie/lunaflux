///|
fn paged_attention_decode_split_cuda_source(
  partial_symbol : String,
  merge_symbol : String,
  query_heads : Int,
  kv_heads : Int,
  head_dimension : Int,
  tokens_per_page : Int,
  page_stride_bytes : Int64,
  physical_page_capacity : Int64,
  max_rows : Int,
  max_tokens : Int,
  max_page_entries : Int,
  workspace : @luna_attention_strategy.AttentionDecodeWorkspacePlan,
) -> Bytes {
  let query_width = query_heads * head_dimension
  let kv_width = kv_heads * head_dimension
  let qkv_width = query_width + 2 * kv_width
  let group_size = query_heads / kv_heads
  let page_stride_values = page_stride_bytes / 2L
  let split_count = workspace.split_count()
  let maximum_context_tokens = workspace.maximum_context_tokens()
  let page_count = 1 + (maximum_context_tokens - 1) / tokens_per_page
  let base_pages_per_split = page_count / split_count
  let larger_split_count = page_count % split_count
  let (maximum_offset_bytes, denominator_offset_bytes, numerator_offset_bytes) = workspace.region_offsets_bytes()
  let maximum_offset = maximum_offset_bytes / 4L
  let denominator_offset = denominator_offset_bytes / 4L
  let numerator_offset = numerator_offset_bytes / 4L
  let accumulator_items = (head_dimension + 31) / 32
  @utf8.encode(
    "#include <cuda_bf16.h>\n" +
    "#include <math_constants.h>\n" +
    "#define LF_BLOCK_THREADS 32\n" +
    "static __device__ __forceinline__ float lf_decode_split_reduce_sum_v1(float value) {\n" +
    "  for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down_sync(0xffffffffU, value, offset);\n" +
    "  return __shfl_sync(0xffffffffU, value, 0);\n" +
    "}\n" +
    "extern \"C\" __global__ void " +
    partial_symbol +
    "(const int *counts, const int *query_positions, const int *query_row_offsets,\n" +
    "    const int *sequence_lengths, const int *page_table_offsets,\n" +
    "    const int *page_table_page_indices, const __nv_bfloat16 *qkv,\n" +
    "    __nv_bfloat16 *key_cache, __nv_bfloat16 *value_cache, float *workspace) {\n" +
    "  const int lane = (int)threadIdx.x, local_row = (int)blockIdx.x;\n" +
    "  const int query_head = (int)blockIdx.y, split = (int)blockIdx.z;\n" +
    "  if (blockDim.x != LF_BLOCK_THREADS || split >= " +
    split_count.to_string() +
    " || query_head >= " +
    query_heads.to_string() +
    ") return;\n" +
    "  const int prefill_count = counts[0], decode_count = counts[1];\n" +
    "  const int row_count = counts[2], token_count = counts[3], page_entry_count = counts[4];\n" +
    "  if (prefill_count < 0 || decode_count <= 0 || row_count != prefill_count + decode_count ||\n" +
    "      row_count > " +
    max_rows.to_string() +
    " || token_count <= 0 || token_count > " +
    max_tokens.to_string() +
    " || page_entry_count <= 0 || page_entry_count > " +
    max_page_entries.to_string() +
    " || local_row >= decode_count) return;\n" +
    "  const int row = prefill_count + local_row;\n" +
    "  const int row_start = query_row_offsets[row], row_end = query_row_offsets[row + 1];\n" +
    "  const unsigned long long partial = ((unsigned long long)row * " +
    query_heads.to_string() +
    "ULL + query_head) * " +
    split_count.to_string() +
    "ULL + split;\n" +
    "  float *partial_maximum = workspace + " +
    maximum_offset.to_string() +
    "ULL;\n" +
    "  float *partial_denominator = workspace + " +
    denominator_offset.to_string() +
    "ULL;\n" +
    "  float *partial_numerator = workspace + " +
    numerator_offset.to_string() +
    "ULL + partial * " +
    head_dimension.to_string() +
    "ULL;\n" +
    "  if (row_start < 0 || row_end != row_start + 1 || row_end > token_count) {\n" +
    "    if (lane == 0) { partial_maximum[partial] = CUDART_NAN_F; partial_denominator[partial] = CUDART_NAN_F; }\n" +
    "    for (int d = lane; d < " +
    head_dimension.to_string() +
    "; d += LF_BLOCK_THREADS) partial_numerator[d] = CUDART_NAN_F;\n" +
    "    return;\n" +
    "  }\n" +
    "  const int token = row_start, query_position = query_positions[token];\n" +
    "  const int page_start = page_table_offsets[row], page_end = page_table_offsets[row + 1];\n" +
    "  if (query_position < 0 || query_position >= sequence_lengths[row] || page_start < 0 ||\n" +
    "      page_start >= page_end || page_end > page_entry_count ||\n" +
    "      page_end - page_start != (sequence_lengths[row] + " +
    (tokens_per_page - 1).to_string() +
    ") / " +
    tokens_per_page.to_string() +
    ") {\n" +
    "    if (lane == 0) { partial_maximum[partial] = CUDART_NAN_F; partial_denominator[partial] = CUDART_NAN_F; }\n" +
    "    for (int d = lane; d < " +
    head_dimension.to_string() +
    "; d += LF_BLOCK_THREADS) partial_numerator[d] = CUDART_NAN_F;\n" +
    "    return;\n" +
    "  }\n" +
    "  const int current_page_slot = page_start + query_position / " +
    tokens_per_page.to_string() +
    ";\n" +
    "  const int current_physical_page = current_page_slot < page_end ? page_table_page_indices[current_page_slot] : -1;\n" +
    "  if (current_physical_page < 0 ||\n" +
    "      (unsigned long long)current_physical_page >= " +
    physical_page_capacity.to_string() +
    "ULL) {\n" +
    "    if (lane == 0) { partial_maximum[partial] = CUDART_NAN_F; partial_denominator[partial] = CUDART_NAN_F; }\n" +
    "    for (int d = lane; d < " +
    head_dimension.to_string() +
    "; d += LF_BLOCK_THREADS) partial_numerator[d] = CUDART_NAN_F;\n" +
    "    return;\n" +
    "  }\n" +
    "  const int partition_begin_page = split * " +
    base_pages_per_split.to_string() +
    " + (split < " +
    larger_split_count.to_string() +
    " ? split : " +
    larger_split_count.to_string() +
    ");\n" +
    "  const int partition_pages = " +
    base_pages_per_split.to_string() +
    " + (split < " +
    larger_split_count.to_string() +
    " ? 1 : 0);\n" +
    "  const int partition_begin = partition_begin_page * " +
    tokens_per_page.to_string() +
    ", partition_limit = (partition_begin_page + partition_pages) * " +
    tokens_per_page.to_string() +
    ";\n" +
    "  const int key_end = query_position + 1 < partition_limit ? query_position + 1 : partition_limit;\n" +
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
    "  if (split == 0 && query_head == 0) {\n" +
    "    const int physical_page = current_physical_page;\n" +
    "    {\n" +
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
    "  float maximum = -CUDART_INF_F, denominator = 0.0f;\n" +
    "  float accumulator[" +
    accumulator_items.to_string() +
    "];\n" +
    "  #pragma unroll\n" +
    "  for (int item = 0; item < " +
    accumulator_items.to_string() +
    "; ++item) accumulator[item] = 0.0f;\n" +
    "  const float scale = rsqrtf((float)" +
    head_dimension.to_string() +
    ");\n" +
    "  for (int key_position = partition_begin; key_position < key_end; ++key_position) {\n" +
    "    const __nv_bfloat16 *key_vector = nullptr, *value_vector = nullptr;\n" +
    "    if (key_position == query_position) {\n" +
    "      const unsigned long long qkv_base = (unsigned long long)token * " +
    qkv_width.to_string() +
    "ULL + (unsigned long long)kv_head * " +
    head_dimension.to_string() +
    "ULL;\n" +
    "      key_vector = qkv + qkv_base + key_qkv_offset; value_vector = qkv + qkv_base + value_qkv_offset;\n" +
    "    } else {\n" +
    "      const int logical_page = key_position / " +
    tokens_per_page.to_string() +
    ", page_slot = page_start + logical_page;\n" +
    "      const int physical_page = page_slot < page_end ? page_table_page_indices[page_slot] : -1;\n" +
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
    "    float dot = 0.0f;\n" +
    "    for (int d = lane; d < " +
    head_dimension.to_string() +
    "; d += LF_BLOCK_THREADS) {\n" +
    "      const float key = key_vector == nullptr ? CUDART_NAN_F : __bfloat162float(key_vector[d]);\n" +
    "      dot += __bfloat162float(qkv[query_offset + d]) * key;\n" +
    "    }\n" +
    "    const float score = lf_decode_split_reduce_sum_v1(dot) * scale;\n" +
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
    accumulator_items.to_string() +
    "; ++item) { const int d = lane + item * 32; if (d < " +
    head_dimension.to_string() +
    ") { const float value = value_vector == nullptr ? CUDART_NAN_F : __bfloat162float(value_vector[d]); accumulator[item] = accumulator[item] * old_scale + new_scale * value; } }\n" +
    "  }\n" +
    "  if (lane == 0) { partial_maximum[partial] = maximum; partial_denominator[partial] = denominator; }\n" +
    "  #pragma unroll\n" +
    "  for (int item = 0; item < " +
    accumulator_items.to_string() +
    "; ++item) { const int d = lane + item * 32; if (d < " +
    head_dimension.to_string() +
    ") partial_numerator[d] = accumulator[item]; }\n" +
    "}\n" +
    "extern \"C\" __global__ void " +
    merge_symbol +
    "(const int *counts, const int *query_row_offsets, const float *workspace, __nv_bfloat16 *output) {\n" +
    "  const int lane = (int)threadIdx.x, local_row = (int)blockIdx.x, query_head = (int)blockIdx.y;\n" +
    "  if (blockDim.x != LF_BLOCK_THREADS || query_head >= " +
    query_heads.to_string() +
    ") return;\n" +
    "  const int prefill_count = counts[0], decode_count = counts[1], row_count = counts[2], token_count = counts[3];\n" +
    "  if (prefill_count < 0 || decode_count <= 0 || token_count <= 0 ||\n" +
    "      row_count != prefill_count + decode_count || row_count > " +
    max_rows.to_string() +
    " || local_row >= decode_count) return;\n" +
    "  const int row = prefill_count + local_row, token = query_row_offsets[row];\n" +
    "  if (token < 0 || token >= token_count || query_row_offsets[row + 1] != token + 1) return;\n" +
    "  const float *partial_maximum = workspace + " +
    maximum_offset.to_string() +
    "ULL, *partial_denominator = workspace + " +
    denominator_offset.to_string() +
    "ULL;\n" +
    "  const float *partial_numerator = workspace + " +
    numerator_offset.to_string() +
    "ULL;\n" +
    "  float maximum = -CUDART_INF_F, denominator = 0.0f; int invalid = 0;\n" +
    "  float accumulator[" +
    accumulator_items.to_string() +
    "];\n" +
    "  #pragma unroll\n" +
    "  for (int item = 0; item < " +
    accumulator_items.to_string() +
    "; ++item) accumulator[item] = 0.0f;\n" +
    "  for (int split = 0; split < " +
    split_count.to_string() +
    "; ++split) {\n" +
    "    const unsigned long long partial = ((unsigned long long)row * " +
    query_heads.to_string() +
    "ULL + query_head) * " +
    split_count.to_string() +
    "ULL + split;\n" +
    "    const float part_maximum = partial_maximum[partial], part_denominator = partial_denominator[partial];\n" +
    "    float old_scale = 1.0f, new_scale = 0.0f;\n" +
    "    if (lane == 0 && (invalid || !isfinite(part_denominator) || (part_denominator > 0.0f && !isfinite(part_maximum)))) {\n" +
    "      invalid = 1; maximum = CUDART_NAN_F; denominator = CUDART_NAN_F; old_scale = CUDART_NAN_F; new_scale = CUDART_NAN_F;\n" +
    "    } else if (lane == 0 && part_denominator > 0.0f) {\n" +
    "      const float next_maximum = denominator == 0.0f ? part_maximum : fmaxf(maximum, part_maximum);\n" +
    "      old_scale = denominator == 0.0f ? 0.0f : expf(maximum - next_maximum);\n" +
    "      new_scale = expf(part_maximum - next_maximum);\n" +
    "      maximum = next_maximum; denominator = denominator * old_scale + part_denominator * new_scale;\n" +
    "    }\n" +
    "    old_scale = __shfl_sync(0xffffffffU, old_scale, 0); new_scale = __shfl_sync(0xffffffffU, new_scale, 0);\n" +
    "    #pragma unroll\n" +
    "    for (int item = 0; item < " +
    accumulator_items.to_string() +
    "; ++item) { const int d = lane + item * 32; if (d < " +
    head_dimension.to_string() +
    ") accumulator[item] = accumulator[item] * old_scale + partial_numerator[partial * " +
    head_dimension.to_string() +
    "ULL + d] * new_scale; }\n" +
    "  }\n" +
    "  denominator = __shfl_sync(0xffffffffU, denominator, 0);\n" +
    "  const unsigned long long output_base = ((unsigned long long)token * " +
    query_heads.to_string() +
    "ULL + query_head) * " +
    head_dimension.to_string() +
    "ULL;\n" +
    "  #pragma unroll\n" +
    "  for (int item = 0; item < " +
    accumulator_items.to_string() +
    "; ++item) { const int d = lane + item * 32; if (d < " +
    head_dimension.to_string() +
    ") output[output_base + d] = __float2bfloat16_rn(denominator > 0.0f ? accumulator[item] / denominator : CUDART_NAN_F); }\n" +
    "}\n",
  )
}
