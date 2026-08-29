///|
fn parallel_cuda_source(
  symbol : String,
  reference : @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering,
  maximum_context_tokens : Int,
) -> Bytes {
  let query_heads = reference.query_heads()
  let kv_heads = reference.kv_heads()
  let head_dimension = reference.head_dimension()
  let tokens_per_page = reference.tokens_per_page()
  let query_width = query_heads * head_dimension
  let kv_width = kv_heads * head_dimension
  let qkv_width = query_width + 2 * kv_width
  let group_size = query_heads / kv_heads
  let page_stride_values = reference.page_stride_bytes() / 2L
  let profile = reference.profile()
  @utf8.encode(
    "#include <cuda_bf16.h>\n" +
    "#include <math_constants.h>\n" +
    "#define LF_BLOCK_THREADS 128\n" +
    "#define LF_MAX_CONTEXT " +
    maximum_context_tokens.to_string() +
    "\n" +
    "static __device__ __forceinline__ float lf_parallel_reduce_sum_v1(float value, float *scratch) {\n" +
    "  const int lane = (int)threadIdx.x; scratch[lane] = value; __syncthreads();\n" +
    "  for (int stride = 64; stride > 0; stride >>= 1) {\n" +
    "    if (lane < stride) scratch[lane] += scratch[lane + stride];\n" +
    "    __syncthreads();\n" +
    "  }\n" +
    "  return scratch[0];\n" +
    "}\n" +
    "static __device__ __forceinline__ float lf_parallel_reduce_max_v1(float value, float *scratch) {\n" +
    "  const int lane = (int)threadIdx.x; scratch[lane] = value; __syncthreads();\n" +
    "  for (int stride = 64; stride > 0; stride >>= 1) {\n" +
    "    if (lane < stride) scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);\n" +
    "    __syncthreads();\n" +
    "  }\n" +
    "  return scratch[0];\n" +
    "}\n" +
    "static __device__ __forceinline__ float lf_parallel_load_v1(\n" +
    "    const __nv_bfloat16 *qkv, const __nv_bfloat16 *cache,\n" +
    "    const int *positions, const int *page_offsets, const int *page_indices,\n" +
    "    int row_start, int row_end, int chunk_start, int key_position,\n" +
    "    int kv_head, int component, int qkv_component_offset) {\n" +
    "  if (key_position >= chunk_start) {\n" +
    "    const int local = row_start + key_position - chunk_start;\n" +
    "    if (local < row_end && positions[local] == key_position) {\n" +
    "      const unsigned long long qkv_index = (unsigned long long)local * " +
    qkv_width.to_string() +
    "ULL + qkv_component_offset + (unsigned long long)kv_head * " +
    head_dimension.to_string() +
    "ULL + component;\n" +
    "      return __bfloat162float(qkv[qkv_index]);\n" +
    "    }\n" +
    "  }\n" +
    "  const int logical_page = key_position / " +
    tokens_per_page.to_string() +
    ";\n" +
    "  const int physical_page = page_indices[page_offsets[0] + logical_page];\n" +
    "  if (physical_page < 0 || (unsigned long long)physical_page >= " +
    reference.physical_page_capacity().to_string() +
    "ULL) return CUDART_NAN_F;\n" +
    "  const int token_in_page = key_position % " +
    tokens_per_page.to_string() +
    ";\n" +
    "  const unsigned long long cache_index = (unsigned long long)physical_page * " +
    page_stride_values.to_string() +
    "ULL + ((unsigned long long)token_in_page * " +
    kv_heads.to_string() +
    "ULL + kv_head) * " +
    head_dimension.to_string() +
    "ULL + component;\n" +
    "  return __bfloat162float(cache[cache_index]);\n" +
    "}\n" +
    "extern \"C\" __global__ void " +
    symbol +
    "(\n" +
    "    const int *counts, const int *query_positions, const int *query_row_offsets,\n" +
    "    const int *sequence_lengths, const int *page_table_offsets,\n" +
    "    const int *page_table_page_indices, const __nv_bfloat16 *qkv,\n" +
    "    __nv_bfloat16 *output, __nv_bfloat16 *key_cache, __nv_bfloat16 *value_cache) {\n" +
    "  const int lane = (int)threadIdx.x;\n" +
    "  const int token = (int)blockIdx.x;\n" +
    "  const int query_head = (int)blockIdx.y;\n" +
    "  if (blockDim.x != LF_BLOCK_THREADS) return;\n" +
    "  const int prefill_count = counts[0], decode_count = counts[1];\n" +
    "  const int row_count = counts[2], token_count = counts[3];\n" +
    "  const int page_entry_count = counts[4];\n" +
    "  if (prefill_count < 0 || decode_count < 0 || row_count <= 0 ||\n" +
    "      row_count != prefill_count + decode_count || row_count > " +
    profile.max_query_rows().to_string() +
    " || token_count <= 0 || token_count > " +
    profile.max_query_tokens().to_string() +
    " || page_entry_count <= 0 || page_entry_count > " +
    profile.max_page_table_entries().to_string() +
    " || token >= token_count || query_head >= " +
    query_heads.to_string() +
    ") return;\n" +
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
    "      page_start < 0 || page_start >= page_end || page_end > page_entry_count ||\n" +
    "      sequence_lengths[row] <= 0 || sequence_lengths[row] > LF_MAX_CONTEXT ||\n" +
    "      page_end - page_start != (sequence_lengths[row] + " +
    (tokens_per_page - 1).to_string() +
    ") / " +
    tokens_per_page.to_string() +
    ") {\n" +
    "    if (lane == 0) output[output_offset] = __float2bfloat16_rn(CUDART_NAN_F);\n" +
    "    return;\n" +
    "  }\n" +
    "  const int query_position = query_positions[token];\n" +
    "  const int chunk_start = query_positions[row_start];\n" +
    "  if (query_position < chunk_start || query_position >= sequence_lengths[row]) {\n" +
    "    if (lane == 0) output[output_offset] = __float2bfloat16_rn(CUDART_NAN_F);\n" +
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
    reference.physical_page_capacity().to_string() +
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
    "  extern __shared__ float shared[];\n" +
    "  float *scores = shared; float *reduction = shared + LF_MAX_CONTEXT;\n" +
    "  const float scale = rsqrtf((float)" +
    head_dimension.to_string() +
    ");\n" +
    "  for (int key_position = 0; key_position <= query_position; ++key_position) {\n" +
    "    float partial = 0.0f;\n" +
    "    for (int d = lane; d < " +
    head_dimension.to_string() +
    "; d += LF_BLOCK_THREADS) {\n" +
    "      const float key = lf_parallel_load_v1(qkv, key_cache, query_positions,\n" +
    "          page_table_offsets + row, page_table_page_indices, row_start, row_end,\n" +
    "          chunk_start, key_position, kv_head, d, key_qkv_offset);\n" +
    "      partial += __bfloat162float(qkv[query_offset + d]) * key;\n" +
    "    }\n" +
    "    const float dot = lf_parallel_reduce_sum_v1(partial, reduction);\n" +
    "    if (lane == 0) scores[key_position] = dot * scale;\n" +
    "    __syncthreads();\n" +
    "  }\n" +
    "  float local_max = -CUDART_INF_F;\n" +
    "  for (int key_position = lane; key_position <= query_position; key_position += LF_BLOCK_THREADS)\n" +
    "    local_max = fmaxf(local_max, scores[key_position]);\n" +
    "  const float maximum = lf_parallel_reduce_max_v1(local_max, reduction);\n" +
    "  float local_sum = 0.0f;\n" +
    "  for (int key_position = lane; key_position <= query_position; key_position += LF_BLOCK_THREADS)\n" +
    "    local_sum += expf(scores[key_position] - maximum);\n" +
    "  const float denominator = lf_parallel_reduce_sum_v1(local_sum, reduction);\n" +
    "  for (int d = lane; d < " +
    head_dimension.to_string() +
    "; d += LF_BLOCK_THREADS) {\n" +
    "    float accumulator = 0.0f;\n" +
    "    for (int key_position = 0; key_position <= query_position; ++key_position) {\n" +
    "      const float value = lf_parallel_load_v1(qkv, value_cache, query_positions,\n" +
    "          page_table_offsets + row, page_table_page_indices, row_start, row_end,\n" +
    "          chunk_start, key_position, kv_head, d, value_qkv_offset);\n" +
    "      accumulator += expf(scores[key_position] - maximum) * value / denominator;\n" +
    "    }\n" +
    "    output[output_offset + d] = __float2bfloat16_rn(accumulator);\n" +
    "  }\n" +
    "}\n",
  )
}
