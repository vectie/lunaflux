///|
fn qwen_fused_source(
  symbol : String,
  semantic : @luna_fusion_plan.QkvQkNormPositionedRopePagedKvWritePlan,
  layout : @device_layout.DeviceKvLayout,
  profile : @launch_contract.PagedKernelProfile,
  dispatch_canary : UInt?,
) -> Bytes {
  let (input_width, output_width) = semantic.projection_shape()
  let (query_heads, kv_heads, head_dimension) = semantic.attention_shape()
  let out = StringBuilder()
  out.write_string(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n",
  )
  out.write_string("#define LF_BLOCK_THREADS 128\n")
  out.write_string("#define LF_INPUT_WIDTH " + input_width.to_string() + "\n")
  out.write_string("#define LF_OUTPUT_WIDTH " + output_width.to_string() + "\n")
  out.write_string("#define LF_QUERY_HEADS " + query_heads.to_string() + "\n")
  out.write_string("#define LF_KV_HEADS " + kv_heads.to_string() + "\n")
  out.write_string("#define LF_HEAD_DIM " + head_dimension.to_string() + "\n")
  out.write_string(
    "#define LF_NORM_EPSILON " +
    @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
      semantic.norm_epsilon(),
    ) +
    "\n",
  )
  out.write_string(
    "#define LF_ROPE_THETA " +
    @luna_cuda_pointwise_aot.canonical_cuda_float_literal(semantic.rope_theta()) +
    "\n",
  )
  out.write_string(
    "#define LF_TOKENS_PER_PAGE " + layout.tokens_per_page().to_string() + "\n",
  )
  out.write_string(
    "#define LF_TOTAL_PAGE_COUNT " +
    layout.total_page_count().to_string() +
    "\n",
  )
  out.write_string(
    "#define LF_PAGE_STRIDE_BF16 " +
    (layout.page_stride_bytes() / 2L).to_string() +
    "LL\n",
  )
  out.write_string("extern \"C\" __global__ void " + symbol + "(\n")
  let canary_argument = match dispatch_canary {
    Some(_) => ",\n  unsigned int* dispatch_canary"
    None => ""
  }
  out.write_string(
    "  const int* step_counts, const int* query_positions,\n" +
    "  const int* query_row_offsets, const int* sequence_lengths,\n" +
    "  const int* page_table_offsets, const int* page_table_page_indices,\n" +
    "  const __nv_bfloat16* input, const __nv_bfloat16* q_weight,\n" +
    "  const __nv_bfloat16* k_weight, const __nv_bfloat16* v_weight,\n" +
    "  const __nv_bfloat16* q_norm_weight,\n" +
    "  const __nv_bfloat16* k_norm_weight, __nv_bfloat16* rotated_qkv,\n" +
    "  __nv_bfloat16* key_cache, __nv_bfloat16* value_cache" +
    canary_argument +
    ") {\n",
  )
  let canary_publication = match dispatch_canary {
    Some(value) =>
      "  if (lane == 0) atomicAdd(dispatch_canary, " +
      value.to_string() +
      "u);\n"
    None => ""
  }
  out.write_string(
    "  if (blockDim.x != LF_BLOCK_THREADS || blockIdx.y != 0 || blockIdx.z != 0) return;\n" +
    "  const int token = (int)blockIdx.x;\n" +
    "  const int prefill = step_counts[0], decode = step_counts[1];\n" +
    "  const int rows = step_counts[2], tokens = step_counts[3];\n" +
    "  const int page_cells = step_counts[4];\n" +
    "  if (prefill < 0 || decode < 0 || rows <= 0 || rows > " +
    profile.max_query_rows().to_string() +
    " || prefill > rows || decode != rows - prefill || tokens <= 0 || tokens > " +
    profile.max_query_tokens().to_string() +
    " || page_cells <= 0 || page_cells > " +
    profile.max_page_table_entries().to_string() +
    " || token >= tokens) return;\n" +
    "  if (query_row_offsets[0] != 0 || query_row_offsets[rows] != tokens ||\n" +
    "      page_table_offsets[0] != 0 || page_table_offsets[rows] != page_cells) return;\n" +
    "  int row = 0;\n" +
    "  while (row + 1 < rows && token >= query_row_offsets[row + 1]) ++row;\n" +
    "  if (token < query_row_offsets[row] || token >= query_row_offsets[row + 1]) return;\n" +
    "  const int position = query_positions[token];\n" +
    "  if (position < 0 || position >= sequence_lengths[row]) return;\n" +
    "  const int logical_page = position / LF_TOKENS_PER_PAGE;\n" +
    "  const int page_begin = page_table_offsets[row], page_end = page_table_offsets[row + 1];\n" +
    "  if (logical_page < 0 || logical_page >= page_end - page_begin) return;\n" +
    "  const int physical_page = page_table_page_indices[page_begin + logical_page];\n" +
    "  if (physical_page < 0 || physical_page >= LF_TOTAL_PAGE_COUNT) return;\n" +
    "  __shared__ float head_values[LF_BLOCK_THREADS];\n" +
    "  __shared__ float reduction[LF_BLOCK_THREADS];\n" +
    "  __shared__ float inverse_rms;\n" +
    "  const int lane = (int)threadIdx.x;\n" +
    "  const int qk_heads = LF_QUERY_HEADS + LF_KV_HEADS;\n" +
    "  const int all_heads = qk_heads + LF_KV_HEADS;\n" +
    "  for (int head = 0; head < all_heads; ++head) {\n" +
    "    const bool is_query = head < LF_QUERY_HEADS;\n" +
    "    const bool is_key = head >= LF_QUERY_HEADS && head < qk_heads;\n" +
    "    const int component = lane;\n" +
    "    if (component < LF_HEAD_DIM) {\n" +
    "      const __nv_bfloat16* matrix = is_query ? q_weight : (is_key ? k_weight : v_weight);\n" +
    "      const int matrix_head = is_query ? head : (is_key ? head - LF_QUERY_HEADS : head - qk_heads);\n" +
    "      const int matrix_column = matrix_head * LF_HEAD_DIM + component;\n" +
    "      float projected = 0.0f;\n" +
    "      for (int inner = 0; inner < LF_INPUT_WIDTH; ++inner) {\n" +
    "        projected = __fadd_rn(projected, __fmul_rn(\n" +
    "          __bfloat162float(input[token * LF_INPUT_WIDTH + inner]),\n" +
    "          __bfloat162float(matrix[matrix_column * LF_INPUT_WIDTH + inner])));\n" +
    "      }\n" +
    "      head_values[component] = __bfloat162float(__float2bfloat16_rn(projected));\n" +
    "    }\n" +
    "    __syncthreads();\n" +
    "    if (head < qk_heads) {\n" +
    "      float square = component < LF_HEAD_DIM ? head_values[component] * head_values[component] : 0.0f;\n" +
    "      reduction[lane] = square;\n" +
    "      __syncthreads();\n" +
    "      for (int stride = LF_BLOCK_THREADS / 2; stride > 0; stride >>= 1) {\n" +
    "        if (lane < stride) reduction[lane] = __fadd_rn(reduction[lane], reduction[lane + stride]);\n" +
    "        __syncthreads();\n" +
    "      }\n" +
    "      if (lane == 0) inverse_rms = rsqrtf(reduction[0] / (float)LF_HEAD_DIM + LF_NORM_EPSILON);\n" +
    "      __syncthreads();\n" +
    "      if (component < LF_HEAD_DIM) {\n" +
    "        const __nv_bfloat16* norm = is_query ? q_norm_weight : k_norm_weight;\n" +
    "        head_values[component] = __bfloat162float(__float2bfloat16_rn(\n" +
    "          head_values[component] * inverse_rms * __bfloat162float(norm[component])));\n" +
    "      }\n" +
    "      __syncthreads();\n" +
    "    }\n" +
    "    if (component < LF_HEAD_DIM) {\n" +
    "      float value = head_values[component];\n" +
    "      if (head < qk_heads) {\n" +
    "        const int half = LF_HEAD_DIM / 2;\n" +
    "        const int paired = component < half ? component + half : component - half;\n" +
    "        const float pair_value = head_values[paired];\n" +
    "        const float exponent = (2.0f * (float)(component % half)) / (float)LF_HEAD_DIM;\n" +
    "        const float angle = (float)position / powf(LF_ROPE_THETA, exponent);\n" +
    "        value = component < half ? value * cosf(angle) - pair_value * sinf(angle) :\n" +
    "          pair_value * sinf(angle) + value * cosf(angle);\n" +
    "      }\n" +
    "      const __nv_bfloat16 encoded = __float2bfloat16_rn(value);\n" +
    "      const int output_column = head * LF_HEAD_DIM + component;\n" +
    "      rotated_qkv[token * LF_OUTPUT_WIDTH + output_column] = encoded;\n" +
    "      if (is_key || head >= qk_heads) {\n" +
    "        const int kv_component = (is_key ? head - LF_QUERY_HEADS : head - qk_heads) * LF_HEAD_DIM + component;\n" +
    "        const long long offset = (long long)physical_page * LF_PAGE_STRIDE_BF16 +\n" +
    "          (long long)(position % LF_TOKENS_PER_PAGE) * (LF_KV_HEADS * LF_HEAD_DIM) + kv_component;\n" +
    "        (is_key ? key_cache : value_cache)[offset] = encoded;\n" +
    "      }\n" +
    "    }\n" +
    "    __syncthreads();\n" +
    "  }\n" +
    canary_publication +
    "}\n",
  )
  @utf8.encode(out.to_string())
}
