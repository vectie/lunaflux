///|
fn readonly_symbol(id : @catalog.AotKernelEntryPointId) -> String {
  "lunaflux_paged_attention_bf16_readonly_v1_ep_" + id.as_int().to_string()
}

///|
/// Emits one four-warp, sixteen-key tile per active query-head tile. The preceding
/// fused ingress kernel has already published every current-token K/V row into
/// the paged cache, so the kernel only reads cache pages. Four independent
/// warps calculate four QK scores concurrently across four warp-local rounds,
/// then the whole block applies one stable online-softmax tile to the output
/// vector. This amortizes block-wide synchronization without increasing the
/// 128-thread launch footprint.
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
  let accumulator_items = (head_dimension + 127) / 128
  let canary_argument = match diagnostic_canary_per_token {
    Some(_) => ",\n    unsigned int *dispatch_canary"
    None => ""
  }
  let canary_publication = match diagnostic_canary_per_token {
    Some(value) =>
      "\n    if (query_head == 0 && thread == 0) dispatch_canary[token] = " +
      value.to_string() +
      "U;"
    None => ""
  }
  @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math_constants.h>\n" +
    "#define LF_READONLY_WARP_THREADS 32\n" +
    "#define LF_READONLY_KEY_WARPS 4\n" +
    "#define LF_READONLY_KEYS_PER_WARP 4\n" +
    "#define LF_READONLY_KEY_TILE 16\n" +
    "#define LF_READONLY_BLOCK_THREADS 128\n" +
    "static __device__ __forceinline__ float lf_readonly_reduce_sum_v4(float value) {\n" +
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
    "  const int thread = (int)threadIdx.x, lane = thread & 31;\n" +
    "  const int warp = thread >> 5, query_head = (int)blockIdx.y;\n" +
    "  if (blockDim.x != LF_READONLY_BLOCK_THREADS || query_head >= " +
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
    "  __shared__ float tile_scores[LF_READONLY_KEY_TILE];\n" +
    "  __shared__ float tile_weights[LF_READONLY_KEY_TILE];\n" +
    "  __shared__ float tile_state[3];\n" +
    "  __shared__ unsigned long long tile_cache_bases[LF_READONLY_KEY_TILE];\n" +
    "  __shared__ int tile_active[LF_READONLY_KEY_TILE];\n" +
    "  __shared__ int tile_valid[LF_READONLY_KEY_TILE];\n" +
    "  __shared__ int tile_ok;\n" +
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
    "      for (int d = thread; d < " +
    head_dimension.to_string() +
    "; d += LF_READONLY_BLOCK_THREADS) output[output_offset + d] = __float2bfloat16_rn(CUDART_NAN_F);\n" +
    "      continue;\n    }\n" +
    "    const int query_position = query_positions[token];\n" +
    "    if (query_position < query_positions[row_start] ||\n" +
    "        query_position >= sequence_lengths[row] ||\n" +
    "        page_end - page_start != (sequence_lengths[row] + " +
    (tokens_per_page - 1).to_string() +
    ") / " +
    tokens_per_page.to_string() +
    ") {\n" +
    "      for (int d = thread; d < " +
    head_dimension.to_string() +
    "; d += LF_READONLY_BLOCK_THREADS) output[output_offset + d] = __float2bfloat16_rn(CUDART_NAN_F);\n" +
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
    accumulator_items.to_string() +
    "];\n" +
    "    #pragma unroll\n" +
    "    for (int item = 0; item < " +
    accumulator_items.to_string() +
    "; ++item) accumulator[item] = 0.0f;\n" +
    "    bool valid = true;\n" +
    "    for (int key_base = 0; key_base <= query_position; key_base += LF_READONLY_KEY_TILE) {\n" +
    "      #pragma unroll\n" +
    "      for (int key_round = 0; key_round < LF_READONLY_KEYS_PER_WARP; ++key_round) {\n" +
    "        const int item = key_round * LF_READONLY_KEY_WARPS + warp;\n" +
    "        const int key_position = key_base + item;\n" +
    "        if (lane == 0) {\n" +
    "          tile_active[item] = key_position <= query_position;\n" +
    "          tile_valid[item] = 1;\n" +
    "          tile_cache_bases[item] = 0ULL;\n" +
    "          if (tile_active[item]) {\n" +
    "          const int logical_page = key_position / " +
    tokens_per_page.to_string() +
    ";\n" +
    "          const int page_slot = page_start + logical_page;\n" +
    "          const int physical_page = page_slot < page_end ? page_table_page_indices[page_slot] : -1;\n" +
    "            if (physical_page < 0 || (unsigned long long)physical_page >= " +
    reference.physical_page_capacity().to_string() +
    "ULL) tile_valid[item] = 0;\n" +
    "            else {\n" +
    "              const int token_in_page = key_position % " +
    tokens_per_page.to_string() +
    ";\n" +
    "              tile_cache_bases[item] = (unsigned long long)physical_page * " +
    page_stride_values.to_string() +
    "ULL + ((unsigned long long)token_in_page * " +
    kv_heads.to_string() +
    "ULL + kv_head) * " +
    head_dimension.to_string() +
    "ULL;\n" +
    "            }\n" +
    "          }\n" +
    "        }\n" +
    "        __syncwarp();\n" +
    "        float partial = 0.0f;\n" +
    "        if (tile_active[item] && tile_valid[item]) {\n" +
    "          const __nv_bfloat16 *key_vector = key_cache + tile_cache_bases[item];\n" +
    "          for (int d = lane; d < " +
    head_dimension.to_string() +
    "; d += LF_READONLY_WARP_THREADS)\n" +
    "            partial += __bfloat162float(rotated_qkv_rows[query_offset + d]) * __bfloat162float(key_vector[d]);\n" +
    "        }\n" +
    "        const float score = lf_readonly_reduce_sum_v4(partial) * scale;\n" +
    "        if (lane == 0) tile_scores[item] = score;\n" +
    "      }\n" +
    "      __syncthreads();\n" +
    "      if (thread == 0) {\n" +
    "        tile_ok = 1;\n" +
    "        float next_maximum = maximum;\n" +
    "        for (int item = 0; item < LF_READONLY_KEY_TILE; ++item) {\n" +
    "          tile_weights[item] = 0.0f;\n" +
    "          if (tile_active[item]) {\n" +
    "            if (!tile_valid[item]) tile_ok = 0;\n" +
    "            else next_maximum = fmaxf(next_maximum, tile_scores[item]);\n" +
    "          }\n" +
    "        }\n" +
    "        const float old_scale = denominator == 0.0f ? 0.0f : expf(maximum - next_maximum);\n" +
    "        float next_denominator = denominator * old_scale;\n" +
    "        if (tile_ok) {\n" +
    "          for (int item = 0; item < LF_READONLY_KEY_TILE; ++item) {\n" +
    "            if (tile_active[item]) {\n" +
    "              tile_weights[item] = expf(tile_scores[item] - next_maximum);\n" +
    "              next_denominator += tile_weights[item];\n" +
    "            }\n" +
    "          }\n" +
    "        }\n" +
    "        tile_state[0] = old_scale;\n" +
    "        tile_state[1] = next_maximum;\n" +
    "        tile_state[2] = next_denominator;\n" +
    "      }\n" +
    "      __syncthreads();\n" +
    "      if (!tile_ok) { valid = false; break; }\n" +
    "      #pragma unroll\n" +
    "      for (int item = 0; item < " +
    accumulator_items.to_string() +
    "; ++item) {\n" +
    "        const int d = thread + item * LF_READONLY_BLOCK_THREADS;\n" +
    "        if (d < " +
    head_dimension.to_string() +
    ") {\n" +
    "          float weighted_value = 0.0f;\n" +
    "          #pragma unroll\n" +
    "          for (int key = 0; key < LF_READONLY_KEY_TILE; ++key)\n" +
    "            if (tile_active[key]) weighted_value += tile_weights[key] * __bfloat162float(value_cache[tile_cache_bases[key] + d]);\n" +
    "          accumulator[item] = accumulator[item] * tile_state[0] + weighted_value;\n" +
    "        }\n" +
    "      }\n" +
    "      maximum = tile_state[1]; denominator = tile_state[2];\n" +
    "      __syncthreads();\n" +
    "    }\n" +
    "    if (!valid || denominator <= 0.0f) {\n" +
    "      for (int d = thread; d < " +
    head_dimension.to_string() +
    "; d += LF_READONLY_BLOCK_THREADS) output[output_offset + d] = __float2bfloat16_rn(CUDART_NAN_F);\n" +
    "      continue;\n" +
    "    }\n" +
    "    #pragma unroll\n" +
    "    for (int item = 0; item < " +
    accumulator_items.to_string() +
    "; ++item) {\n" +
    "      const int d = thread + item * LF_READONLY_BLOCK_THREADS;\n" +
    "      if (d < " +
    head_dimension.to_string() +
    ") output[output_offset + d] = __float2bfloat16_rn(accumulator[item] / denominator);\n" +
    "    }" +
    canary_publication +
    "\n    __syncthreads();\n" +
    "  }\n}\n",
  )
}
