///|
fn qwen_fused_source(
  symbol : String,
  semantic : @luna_fusion_plan.QkvQkNormPositionedRopePagedKvWritePlan,
  layout : @device_layout.DeviceKvLayout,
  profile : @launch_contract.PagedKernelProfile,
  tile_compilation : @luna_projection_tile_compiler.ScheduledProjectionTileCompilation,
  dispatch_canary : UInt?,
) -> Bytes {
  let (input_width, output_width) = semantic.projection_shape()
  let (query_heads, kv_heads, head_dimension) = semantic.attention_shape()
  let loops = tile_compilation.schedule().loops()
  guard loops.get(0) is Some(row_loop) &&
    loops.get(1) is Some(column_loop) &&
    loops.get(2) is Some(reduction_loop) else {
    abort("attention ingress tile schedule lacks projection loops")
  }
  let (_, _, _, tile_rows) = row_loop.components()
  let (_, _, _, tile_columns) = column_loop.components()
  let (_, _, _, tile_reduction) = reduction_loop.components()
  if !tile_compilation.schedule().fuses_attention_ingress_epilogue() ||
    tile_rows != 16 ||
    tile_columns != 16 ||
    tile_reduction != 16 ||
    head_dimension > 128 ||
    head_dimension % 16 != 0 {
    abort("CUDA attention ingress lowering requires a 16x16x16 schedule")
  }
  let packed_heads = query_heads + 2 * kv_heads
  let qk_heads = query_heads + kv_heads
  let kv_width = kv_heads * head_dimension
  let maximum_tile_count = (profile.max_query_tokens() + tile_rows - 1) /
    tile_rows
  let out = StringBuilder()
  out.write_string(
    "#include <cuda_bf16.h>\n#include <mma.h>\n#include <math.h>\n#include <stdint.h>\n",
  )
  out.write_string("#define LF_BLOCK_THREADS 128\n")
  out.write_string("#define LF_WARPS 4\n")
  out.write_string("#define LF_TILE_ROWS " + tile_rows.to_string() + "\n")
  out.write_string("#define LF_INPUT_WIDTH " + input_width.to_string() + "\n")
  out.write_string("#define LF_OUTPUT_WIDTH " + output_width.to_string() + "\n")
  out.write_string("#define LF_QUERY_HEADS " + query_heads.to_string() + "\n")
  out.write_string("#define LF_KV_HEADS " + kv_heads.to_string() + "\n")
  out.write_string("#define LF_QK_HEADS " + qk_heads.to_string() + "\n")
  out.write_string("#define LF_PACKED_HEADS " + packed_heads.to_string() + "\n")
  out.write_string("#define LF_HEAD_DIM " + head_dimension.to_string() + "\n")
  out.write_string("#define LF_KV_WIDTH " + kv_width.to_string() + "\n")
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
    "ULL\n",
  )
  out.write_string("extern \"C\" __global__ void " + symbol + "(\n")
  let canary_argument = match dispatch_canary {
    Some(_) => ",\n  unsigned int* dispatch_canary"
    None => ""
  }
  out.write_string(
    "  const int32_t* __restrict__ step_counts, const int32_t* __restrict__ query_positions,\n" +
    "  const int32_t* __restrict__ query_row_offsets, const int32_t* __restrict__ sequence_lengths,\n" +
    "  const int32_t* __restrict__ page_table_offsets, const int32_t* __restrict__ page_table_page_indices,\n" +
    "  const __nv_bfloat16* __restrict__ input, const __nv_bfloat16* __restrict__ q_weight,\n" +
    "  const __nv_bfloat16* __restrict__ k_weight, const __nv_bfloat16* __restrict__ v_weight,\n" +
    "  const __nv_bfloat16* __restrict__ q_norm_weight, const __nv_bfloat16* __restrict__ k_norm_weight,\n" +
    "  __nv_bfloat16* __restrict__ rotated_qkv, __nv_bfloat16* __restrict__ key_cache,\n" +
    "  __nv_bfloat16* __restrict__ value_cache" +
    canary_argument +
    ") {\n",
  )
  out.write_string(
    "  if (blockDim.x != LF_BLOCK_THREADS || blockIdx.z != 0 || gridDim.x == 0 || gridDim.x > " +
    maximum_tile_count.to_string() +
    " || gridDim.y != LF_PACKED_HEADS) return;\n" +
    "  const int lane = (int)threadIdx.x & 31;\n" +
    "  const int warp = (int)threadIdx.x >> 5;\n" +
    "  const int head = (int)blockIdx.y;\n" +
    "  const int token_begin = (int)blockIdx.x * LF_TILE_ROWS;\n" +
    "  const int prefill = step_counts[0], decode = step_counts[1];\n" +
    "  const int rows = step_counts[2], tokens = step_counts[3], table_cells = step_counts[4];\n" +
    "  if (prefill < 0 || decode < 0 || rows <= 0 || rows > " +
    profile.max_query_rows().to_string() +
    " || decode != rows - prefill || tokens <= 0 || tokens > " +
    profile.max_query_tokens().to_string() +
    " || table_cells <= 0 || table_cells > " +
    profile.max_page_table_entries().to_string() +
    " || token_begin >= tokens || query_row_offsets[0] != 0 ||\n" +
    "      query_row_offsets[rows] != tokens || page_table_offsets[0] != 0 ||\n" +
    "      page_table_offsets[rows] != table_cells) return;\n" +
    "  const bool is_query = head < LF_QUERY_HEADS;\n" +
    "  const bool is_key = head >= LF_QUERY_HEADS && head < LF_QK_HEADS;\n" +
    "  const int matrix_head = is_query ? head : (is_key ? head - LF_QUERY_HEADS : head - LF_QK_HEADS);\n" +
    "  const __nv_bfloat16* matrix = is_query ? q_weight : (is_key ? k_weight : v_weight);\n" +
    "  __shared__ float projected[LF_TILE_ROWS][LF_HEAD_DIM];\n" +
    "  if (tokens == 1 || token_begin + LF_TILE_ROWS > tokens) {\n" +
    "    for (int local_row = warp; local_row < LF_TILE_ROWS; local_row += LF_WARPS) {\n" +
    "      const int token = token_begin + local_row;\n" +
    "      if (token < tokens) for (int component = lane; component < LF_HEAD_DIM; component += 32) {\n" +
    "        const int column = matrix_head * LF_HEAD_DIM + component;\n" +
    "        float value = 0.0f;\n" +
    "        for (int inner = 0; inner < LF_INPUT_WIDTH; ++inner) value = __fadd_rn(value, __fmul_rn(\n" +
    "          __bfloat162float(input[(unsigned long long)token * LF_INPUT_WIDTH + inner]),\n" +
    "          __bfloat162float(matrix[(unsigned long long)column * LF_INPUT_WIDTH + inner])));\n" +
    "        projected[local_row][component] = __bfloat162float(__float2bfloat16_rn(value));\n" +
    "      }\n" +
    "    }\n" +
    "  } else {\n" +
    "    for (int column_base = warp * 16; column_base < LF_HEAD_DIM; column_base += LF_WARPS * 16) {\n" +
    "      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> a;\n" +
    "      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::col_major> b;\n" +
    "      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c;\n" +
    "      nvcuda::wmma::fill_fragment(c, 0.0f);\n" +
    "      for (int inner = 0; inner < LF_INPUT_WIDTH; inner += 16) {\n" +
    "        nvcuda::wmma::load_matrix_sync(a, input + (unsigned long long)token_begin * LF_INPUT_WIDTH + inner, LF_INPUT_WIDTH);\n" +
    "        nvcuda::wmma::load_matrix_sync(b, matrix + (unsigned long long)(matrix_head * LF_HEAD_DIM + column_base) * LF_INPUT_WIDTH + inner, LF_INPUT_WIDTH);\n" +
    "        nvcuda::wmma::mma_sync(c, a, b, c);\n" +
    "      }\n" +
    "      nvcuda::wmma::store_matrix_sync(&projected[0][column_base], c, LF_HEAD_DIM, nvcuda::wmma::mem_row_major);\n" +
    "    }\n" +
    "  }\n" +
    "  __syncthreads();\n",
  )
  out.write_string(
    "  for (int local_row = warp; local_row < LF_TILE_ROWS; local_row += LF_WARPS) {\n" +
    "    const int token = token_begin + local_row;\n" +
    "    if (token >= tokens) continue;\n" +
    "    int row = 0;\n" +
    "    if (lane == 0) while (row + 1 < rows && token >= query_row_offsets[row + 1]) ++row;\n" +
    "    row = __shfl_sync(0xffffffffU, row, 0);\n" +
    "    const int position = query_positions[token];\n" +
    "    const int page_begin = page_table_offsets[row], page_end = page_table_offsets[row + 1];\n" +
    "    const int logical_page = position / LF_TOKENS_PER_PAGE;\n" +
    "    if (row < 0 || row >= rows || token < query_row_offsets[row] || token >= query_row_offsets[row + 1] ||\n" +
    "        position < 0 || position >= sequence_lengths[row] || page_begin < 0 || page_begin >= page_end ||\n" +
    "        page_end > table_cells || logical_page < 0 || logical_page >= page_end - page_begin) continue;\n" +
    "    const int physical_page = page_table_page_indices[page_begin + logical_page];\n" +
    "    if (physical_page < 0 || physical_page >= LF_TOTAL_PAGE_COUNT) continue;\n" +
    "    for (int component = lane; component < LF_HEAD_DIM; component += 32) projected[local_row][component] =\n" +
    "      __bfloat162float(__float2bfloat16_rn(projected[local_row][component]));\n" +
    "    __syncwarp();\n" +
    "    if (head < LF_QK_HEADS) {\n" +
    "      float square_sum = 0.0f;\n" +
    "      for (int component = lane; component < LF_HEAD_DIM; component += 32) square_sum = __fadd_rn(\n" +
    "        square_sum, __fmul_rn(projected[local_row][component], projected[local_row][component]));\n" +
    "      for (int delta = 16; delta > 0; delta >>= 1) square_sum = __fadd_rn(\n" +
    "        square_sum, __shfl_down_sync(0xffffffffU, square_sum, delta));\n" +
    "      const float inverse_rms = __shfl_sync(0xffffffffU,\n" +
    "        lane == 0 ? rsqrtf(__fadd_rn(__fdividef(square_sum, (float)LF_HEAD_DIM), LF_NORM_EPSILON)) : 0.0f, 0);\n" +
    "      const __nv_bfloat16* norm_weight = is_query ? q_norm_weight : k_norm_weight;\n" +
    "      for (int component = lane; component < LF_HEAD_DIM; component += 32) projected[local_row][component] =\n" +
    "        __bfloat162float(__float2bfloat16_rn(__fmul_rn(__fmul_rn(projected[local_row][component], inverse_rms),\n" +
    "        __bfloat162float(norm_weight[component]))));\n" +
    "      __syncwarp();\n" +
    "    }\n" +
    "    for (int component = lane; component < LF_HEAD_DIM; component += 32) {\n" +
    "      float value = projected[local_row][component];\n" +
    "      if (head < LF_QK_HEADS) {\n" +
    "        const int half = LF_HEAD_DIM / 2, pair = component % half;\n" +
    "        const int paired = component < half ? component + half : component - half;\n" +
    "        const float angle = (float)position / powf(LF_ROPE_THETA, (2.0f * (float)pair) / (float)LF_HEAD_DIM);\n" +
    "        const float cosine = cosf(angle), sine = sinf(angle);\n" +
    "        value = component < half ? __fsub_rn(__fmul_rn(value, cosine),\n" +
    "          __fmul_rn(projected[local_row][paired], sine)) : __fadd_rn(\n" +
    "          __fmul_rn(projected[local_row][paired], sine), __fmul_rn(value, cosine));\n" +
    "      }\n" +
    "      const __nv_bfloat16 encoded = __float2bfloat16_rn(value);\n" +
    "      const unsigned long long output_index = ((unsigned long long)token * LF_PACKED_HEADS + head) * LF_HEAD_DIM + component;\n" +
    "      rotated_qkv[output_index] = encoded;\n" +
    "      if (head >= LF_QUERY_HEADS) {\n" +
    "        const int kv_head = is_key ? head - LF_QUERY_HEADS : head - LF_QK_HEADS;\n" +
    "        const unsigned long long cache_index = (unsigned long long)physical_page * LF_PAGE_STRIDE_BF16 +\n" +
    "          (unsigned long long)(position % LF_TOKENS_PER_PAGE) * LF_KV_WIDTH +\n" +
    "          (unsigned long long)kv_head * LF_HEAD_DIM + component;\n" +
    "        (is_key ? key_cache : value_cache)[cache_index] = encoded;\n" +
    "      }\n" +
    "    }\n" +
    "    __syncwarp();\n" +
    "  }\n",
  )
  match dispatch_canary {
    Some(value) =>
      out.write_string(
        "  if (threadIdx.x == 0 && head == 0) atomicAdd(dispatch_canary, " +
        value.to_string() +
        "u);\n",
      )
    None => ()
  }
  out.write_string("}\n")
  @utf8.encode(out.to_string())
}
