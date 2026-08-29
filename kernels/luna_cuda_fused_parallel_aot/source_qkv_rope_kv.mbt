///|
fn fused_qkv_rope_kv_source(
  symbol : String,
  parameters : FusedQkvParameters,
  profile : @launch_contract.PagedKernelProfile,
  diagnostic_canary : UInt?,
) -> Bytes {
  let output = StringBuilder()
  let canary_argument = match diagnostic_canary {
    Some(_) => ", unsigned int* dispatch_canary"
    None => ""
  }
  output.write_string(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n",
  )
  output.write_string("#define LF_BLOCK_THREADS 128\n")
  output.write_string(
    "#define LF_INPUT_WIDTH " + parameters.input_width.to_string() + "\n",
  )
  output.write_string(
    "#define LF_OUTPUT_WIDTH " + parameters.output_width.to_string() + "\n",
  )
  output.write_string(
    "#define LF_QUERY_HEADS " + parameters.query_heads.to_string() + "\n",
  )
  output.write_string(
    "#define LF_KV_HEADS " + parameters.kv_heads.to_string() + "\n",
  )
  output.write_string(
    "#define LF_HEAD_DIM " + parameters.head_dimension.to_string() + "\n",
  )
  output.write_string(
    "#define LF_TOKENS_PER_PAGE " +
    parameters.tokens_per_page.to_string() +
    "\n",
  )
  output.write_string(
    "#define LF_TOTAL_PAGE_COUNT " +
    parameters.total_page_count.to_string() +
    "\n",
  )
  output.write_string(
    "#define LF_PAGE_STRIDE_BF16 " +
    (parameters.page_stride_bytes / 2L).to_string() +
    "LL\n",
  )
  output.write_string(
    "#define LF_ROPE_THETA " +
    @luna_cuda_pointwise_aot.canonical_cuda_float_literal(parameters.rope_theta) +
    "\n",
  )
  output.write_string("extern \"C\" __global__ void " + symbol + "(\n")
  output.write_string(
    "  const int* step_counts, const int* query_positions,\n" +
    "  const int* query_row_offsets, const int* sequence_lengths,\n" +
    "  const int* page_table_offsets, const int* page_table_page_indices,\n" +
    "  const __nv_bfloat16* input, const __nv_bfloat16* q_weight,\n" +
    "  const __nv_bfloat16* k_weight, const __nv_bfloat16* v_weight,\n" +
    "  __nv_bfloat16* rotated_qkv, __nv_bfloat16* key_cache,\n" +
    "  __nv_bfloat16* value_cache" +
    canary_argument +
    ") {\n",
  )
  let canary_publication = match diagnostic_canary {
    Some(canary) =>
      "  __syncthreads();\n  if (threadIdx.x == 0) atomicAdd(dispatch_canary, " +
      canary.to_string() +
      "u);\n"
    None => ""
  }
  output.write_string(
    "  if (blockDim.x != LF_BLOCK_THREADS || blockIdx.y != 0 || blockIdx.z != 0) return;\n" +
    "  const int token = (int)blockIdx.x;\n" +
    "  const int prefill = step_counts[0], decode = step_counts[1];\n" +
    "  const int rows = step_counts[2], tokens = step_counts[3];\n" +
    "  const int page_cells = step_counts[4];\n" +
    "  if (prefill < 0 || decode < 0 || rows <= 0 || rows > " +
    profile.max_query_rows().to_string() +
    " || prefill > rows || decode != rows - prefill ||\n" +
    "      tokens <= 0 || tokens > " +
    profile.max_query_tokens().to_string() +
    " ||\n" +
    "      page_cells <= 0 || page_cells > " +
    profile.max_page_table_entries().to_string() +
    " || token >= tokens) return;\n" +
    "  if (query_row_offsets[0] != 0 || query_row_offsets[rows] != tokens ||\n" +
    "      page_table_offsets[0] != 0 || page_table_offsets[rows] != page_cells) return;\n" +
    "  for (int r = 0; r < rows; ++r) {\n" +
    "    if (query_row_offsets[r] < 0 || query_row_offsets[r] >= query_row_offsets[r + 1] ||\n" +
    "        query_row_offsets[r + 1] > tokens || sequence_lengths[r] <= 0 ||\n" +
    "        page_table_offsets[r] < 0 || page_table_offsets[r] >= page_table_offsets[r + 1] ||\n" +
    "        page_table_offsets[r + 1] > page_cells) return;\n" +
    "  }\n" +
    "  int row = 0;\n" +
    "  while (row + 1 < rows && token >= query_row_offsets[row + 1]) ++row;\n" +
    "  if (token < query_row_offsets[row] || token >= query_row_offsets[row + 1]) return;\n" +
    "  const int local_token = token - query_row_offsets[row];\n" +
    "  const int position = query_positions[token];\n" +
    "  if (position < 0 || position >= sequence_lengths[row] || local_token < 0) return;\n" +
    "  const int logical_page = position / LF_TOKENS_PER_PAGE;\n" +
    "  const int page_begin = page_table_offsets[row], page_end = page_table_offsets[row + 1];\n" +
    "  if (logical_page < 0 || logical_page >= page_end - page_begin) return;\n" +
    "  const int page_slot = page_begin + logical_page;\n" +
    "  const int physical_page = page_table_page_indices[page_slot];\n" +
    "  if (physical_page < 0 || physical_page >= LF_TOTAL_PAGE_COUNT) return;\n",
  )
  output.write_string(
    "  for (int column = (int)threadIdx.x; column < LF_OUTPUT_WIDTH; column += LF_BLOCK_THREADS) {\n" +
    "    const __nv_bfloat16* weight = column < LF_QUERY_HEADS * LF_HEAD_DIM ? q_weight :\n" +
    "      (column < (LF_QUERY_HEADS + LF_KV_HEADS) * LF_HEAD_DIM ? k_weight : v_weight);\n" +
    "    const int weight_column = column < LF_QUERY_HEADS * LF_HEAD_DIM ? column :\n" +
    "      (column < (LF_QUERY_HEADS + LF_KV_HEADS) * LF_HEAD_DIM ?\n" +
    "       column - LF_QUERY_HEADS * LF_HEAD_DIM :\n" +
    "       column - (LF_QUERY_HEADS + LF_KV_HEADS) * LF_HEAD_DIM);\n" +
    "    float projected = 0.0f;\n" +
    "    for (int inner = 0; inner < LF_INPUT_WIDTH; ++inner) {\n" +
    "      const float product = __fmul_rn(__bfloat162float(input[token * LF_INPUT_WIDTH + inner]),\n" +
    "        __bfloat162float(weight[weight_column * LF_INPUT_WIDTH + inner]));\n" +
    "      projected = __fadd_rn(projected, product);\n" +
    "    }\n" +
    "    projected = __bfloat162float(__float2bfloat16_rn(projected));\n" +
    "    float rotated = projected;\n" +
    "    const int qk_width = (LF_QUERY_HEADS + LF_KV_HEADS) * LF_HEAD_DIM;\n" +
    "    if (column < qk_width) {\n" +
    "      const int component = column % LF_HEAD_DIM;\n" +
    "      const int half = LF_HEAD_DIM / 2;\n" +
    "      const int paired_component = component < half ? component + half : component - half;\n" +
    "      const int paired_column = column - component + paired_component;\n" +
    "      const __nv_bfloat16* paired_weight = column < LF_QUERY_HEADS * LF_HEAD_DIM ? q_weight : k_weight;\n" +
    "      const int paired_weight_column = column < LF_QUERY_HEADS * LF_HEAD_DIM ? paired_column : paired_column - LF_QUERY_HEADS * LF_HEAD_DIM;\n" +
    "      float paired = 0.0f;\n" +
    "      for (int inner = 0; inner < LF_INPUT_WIDTH; ++inner) {\n" +
    "        const float paired_product = __fmul_rn(__bfloat162float(input[token * LF_INPUT_WIDTH + inner]),\n" +
    "          __bfloat162float(paired_weight[paired_weight_column * LF_INPUT_WIDTH + inner]));\n" +
    "        paired = __fadd_rn(paired, paired_product);\n" +
    "      }\n" +
    "      paired = __bfloat162float(__float2bfloat16_rn(paired));\n" +
    "      const float exponent = (2.0f * (float)(component % half)) / (float)LF_HEAD_DIM;\n" +
    "      const float angle = (float)position / powf(LF_ROPE_THETA, exponent);\n" +
    "      rotated = component < half ? projected * cosf(angle) - paired * sinf(angle) :\n" +
    "        paired * sinf(angle) + projected * cosf(angle);\n" +
    "    }\n" +
    "    const __nv_bfloat16 encoded = __float2bfloat16_rn(rotated);\n" +
    "    rotated_qkv[token * LF_OUTPUT_WIDTH + column] = encoded;\n",
  )
  output.write_string(
    "    const int key_begin = LF_QUERY_HEADS * LF_HEAD_DIM;\n" +
    "    const int value_begin = (LF_QUERY_HEADS + LF_KV_HEADS) * LF_HEAD_DIM;\n" +
    "    if (column >= key_begin) {\n" +
    "      const bool is_value = column >= value_begin;\n" +
    "      const int kv_component = column - (is_value ? value_begin : key_begin);\n" +
    "      const long long cache_offset = (long long)physical_page * LF_PAGE_STRIDE_BF16 +\n" +
    "        (long long)(position % LF_TOKENS_PER_PAGE) * (LF_KV_HEADS * LF_HEAD_DIM) + kv_component;\n" +
    "      (is_value ? value_cache : key_cache)[cache_offset] = encoded;\n" +
    "    }\n" +
    "  }\n" +
    canary_publication +
    "}\n",
  )
  @utf8.encode(output.to_string())
}
