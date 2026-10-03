///|
fn qwen_fused_source(
  symbol : String,
  semantic : @luna_fusion_plan.QkvQkNormPositionedRopePagedKvWritePlan,
  layout : @device_layout.DeviceKvLayout,
  profile : @launch_contract.PagedKernelProfile,
  tile_compilation : @luna_projection_tile_compiler.ScheduledProjectionTileCompilation,
  physical : QwenIngressPhysicalProgram,
  dispatch_canary : UInt?,
  step_rotary_cache? : Bool = false,
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
    !tile_compilation.schedule().hoists_rotary_inverse_frequencies() ||
    !tile_compilation.schedule().fuses_rotary_pairs() ||
    tile_compilation.schedule().row_tail_policy() != ZeroPaddedMatrixRowTail ||
    tile_compilation.schedule().single_row_reduction() != StridedPairwiseDot(32) ||
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
  let tile_rows = physical.pipeline.schedule.row_width()
  let (head_groups, heads_per_group, projection_width, _) = physical.packing.geometry()
  // This backend implements one row-scoped rotary lifetime around all local
  // complete-head consumers. Reject new effect orders rather than silently
  // emitting this lowering for an incompatible middle-end program.
  let effects = physical.packing.effects()
  guard effects.length() == 3 &&
    effects[0] == PrepareRowRotary &&
    effects[1] == ConsumeCompleteHeads &&
    effects[2] == RetireRowRotary else {
    abort("CUDA ingress does not support this rotary lifetime")
  }
  let maximum_tile_count = (profile.max_query_tokens() + tile_rows - 1) /
    tile_rows
  let column_plan = physical.columns
  let (consumer_groups, column_rounds, live_columns, column_windows) = column_plan.geometry()
  let out = StringBuilder()
  out.write_string(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n",
  )
  out.write_string("#define LF_BLOCK_THREADS 128\n")
  out.write_string("#define LF_WARPS \{consumer_groups}\n")
  out.write_string("#define LF_COLUMN_ROUNDS \{column_rounds}\n")
  out.write_string("#define LF_LIVE_COLUMNS \{live_columns}\n")
  out.write_string("#define LF_COLUMN_WINDOWS \{column_windows}\n")
  out.write_string("#define LF_TILE_ROWS " + tile_rows.to_string() + "\n")
  out.write_string("#define LF_INPUT_WIDTH " + input_width.to_string() + "\n")
  out.write_string("#define LF_OUTPUT_WIDTH " + output_width.to_string() + "\n")
  out.write_string("#define LF_QUERY_HEADS " + query_heads.to_string() + "\n")
  out.write_string("#define LF_KV_HEADS " + kv_heads.to_string() + "\n")
  out.write_string("#define LF_QK_HEADS " + qk_heads.to_string() + "\n")
  out.write_string("#define LF_PACKED_HEADS " + packed_heads.to_string() + "\n")
  out.write_string("#define LF_HEAD_GROUPS \{head_groups}\n")
  out.write_string("#define LF_HEADS_PER_CTA \{heads_per_group}\n")
  out.write_string("#define LF_PROJECTION_WIDTH \{projection_width}\n")
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
    if step_rotary_cache {
      @luna_cuda_attention_ingress_source.lower_cached_attention_ingress_numerics(
        physical.numerics,
      )
    } else {
      @luna_cuda_attention_ingress_source.lower_prepared_attention_ingress_numerics(
        physical.numerics,
      )
    },
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
    (if step_rotary_cache {
      ",\n  const float2* __restrict__ rotary_pairs"
    } else {
      ""
    }) +
    canary_argument +
    ") {\n",
  )
  out.write_string(
    "  if (blockDim.x != LF_BLOCK_THREADS || blockIdx.z != 0 || gridDim.x == 0 || gridDim.x > " +
    maximum_tile_count.to_string() +
    " || gridDim.y != LF_HEAD_GROUPS) return;\n" +
    "  const int lane = (int)threadIdx.x & 31;\n" +
    "  const int warp = (int)threadIdx.x >> 5;\n" +
    "  const int head_begin = (int)blockIdx.y * LF_HEADS_PER_CTA;\n" +
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
    (if physical.scratch.allocation_bytes() > 49152L {
      "  extern __shared__ __align__(32) unsigned char ingress_scratch[];\n" +
      "  float (*projected)[LF_PROJECTION_WIDTH] = reinterpret_cast<float (*)[LF_PROJECTION_WIDTH]>(ingress_scratch + \{if physical.scratch.reuses_storage() { 0L } else { physical.pipeline.operands.allocation_elements().to_int64() * 2L }});\n"
    } else if physical.scratch.reuses_storage() {
      "  __shared__ __align__(32) unsigned char ingress_scratch[\{physical.scratch.allocation_bytes()}];\n" +
      "  float (*projected)[LF_PROJECTION_WIDTH] = reinterpret_cast<float (*)[LF_PROJECTION_WIDTH]>(ingress_scratch);\n"
    } else {
      "  __shared__ float projected[LF_TILE_ROWS][LF_PROJECTION_WIDTH];\n"
    }) +
    "  if (tokens == 1) {\n" +
    "    // Lower the compiler's strided fold followed by a fixed pairwise tree.\n" +
    "    for (int component = warp; component < LF_PROJECTION_WIDTH; component += LF_WARPS) {\n" +
    "      const int head = head_begin + component / LF_HEAD_DIM;\n" +
    "      if (head >= LF_PACKED_HEADS) continue;\n" +
    "      const bool is_query = head < LF_QUERY_HEADS, is_key = head < LF_QK_HEADS;\n" +
    "      const int matrix_head = is_query ? head : (is_key ? head - LF_QUERY_HEADS : head - LF_QK_HEADS);\n" +
    "      const __nv_bfloat16* matrix = is_query ? q_weight : (is_key ? k_weight : v_weight);\n" +
    "      const int column = matrix_head * LF_HEAD_DIM + component % LF_HEAD_DIM;\n" +
    "      float value = 0.0f;\n" +
    "      for (int inner = lane; inner < LF_INPUT_WIDTH; inner += \{physical.scalar.geometry().2}) value = __fadd_rn(value, __fmul_rn(\n" +
    "        __bfloat162float(input[inner]),\n" +
    "        __bfloat162float(matrix[(unsigned long long)column * LF_INPUT_WIDTH + inner])));\n" +
    "      for (int offset = \{physical.scalar.combination_distances()[0]}; offset > 0; offset >>= 1)\n" +
    "        value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, offset));\n" +
    "      if (lane == 0) projected[0][component] = __bfloat162float(__float2bfloat16_rn(value));\n" +
    "    }\n" +
    "  } else {\n" +
    ingress_matrix_projection_source(physical) +
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
    "    // Row-scoped rotary operands remain live until every owned Q/K head retires.\n" +
    "    LfIngressRotaryValues rotary;\n" +
    (if step_rotary_cache {
      "    if (head_begin < LF_QK_HEADS) rotary = lf_ingress_load_rotary(rotary_pairs, token, lane);\n"
    } else {
      "    if (head_begin < LF_QK_HEADS) rotary = lf_ingress_prepare_rotary(position, lane);\n"
    }) +
    "    #pragma unroll\n" +
    "    for (int local_head = 0; local_head < LF_HEADS_PER_CTA; ++local_head) {\n" +
    "    const int head = head_begin + local_head;\n" +
    "    if (head >= LF_PACKED_HEADS) continue;\n" +
    "    const bool is_query = head < LF_QUERY_HEADS;\n" +
    "    const bool is_key = head >= LF_QUERY_HEADS && head < LF_QK_HEADS;\n" +
    "    float* head_values = projected[local_row] + local_head * LF_HEAD_DIM;\n" +
    "    for (int component = lane; component < LF_HEAD_DIM; component += 32) head_values[component] =\n" +
    "      __bfloat162float(__float2bfloat16_rn(head_values[component]));\n" +
    (if physical.numerics.pair_retention() is Some(_) {
      ""
    } else {
      "    __syncwarp();\n"
    }) +
    "    if (head < LF_QK_HEADS) {\n" +
    "      const __nv_bfloat16* norm_weight = is_query ? q_norm_weight : k_norm_weight;\n" +
    "      lf_ingress_norm_rotary_prepared(head_values, norm_weight, rotary, lane);\n" +
    "    }\n" +
    "    if (head < LF_QK_HEADS) {\n" +
    "      const int half = LF_HEAD_DIM / 2;\n" +
    "      for (int pair = lane; pair < half; pair += 32) {\n" +
    "        const __nv_bfloat16 first_encoded = __float2bfloat16_rn(head_values[pair]);\n" +
    "        const __nv_bfloat16 second_encoded = __float2bfloat16_rn(head_values[pair + half]);\n" +
    "        const unsigned long long output_base =\n" +
    "          ((unsigned long long)token * LF_PACKED_HEADS + head) * LF_HEAD_DIM;\n" +
    "        rotated_qkv[output_base + pair] = first_encoded;\n" +
    "        rotated_qkv[output_base + pair + half] = second_encoded;\n" +
    "        if (is_key) {\n" +
    "          const int kv_head = head - LF_QUERY_HEADS;\n" +
    "          const unsigned long long cache_base =\n" +
    "            (unsigned long long)physical_page * LF_PAGE_STRIDE_BF16 +\n" +
    "            (unsigned long long)(position % LF_TOKENS_PER_PAGE) * LF_KV_WIDTH +\n" +
    "            (unsigned long long)kv_head * LF_HEAD_DIM;\n" +
    "          key_cache[cache_base + pair] = first_encoded;\n" +
    "          key_cache[cache_base + pair + half] = second_encoded;\n" +
    "        }\n" +
    "      }\n" +
    "    } else {\n" +
    "      const int kv_head = head - LF_QK_HEADS;\n" +
    "      for (int component = lane; component < LF_HEAD_DIM; component += 32) {\n" +
    "        const __nv_bfloat16 encoded =\n" +
    "          __float2bfloat16_rn(head_values[component]);\n" +
    "        const unsigned long long output_index =\n" +
    "          ((unsigned long long)token * LF_PACKED_HEADS + head) * LF_HEAD_DIM + component;\n" +
    "        const unsigned long long cache_index =\n" +
    "          (unsigned long long)physical_page * LF_PAGE_STRIDE_BF16 +\n" +
    "          (unsigned long long)(position % LF_TOKENS_PER_PAGE) * LF_KV_WIDTH +\n" +
    "          (unsigned long long)kv_head * LF_HEAD_DIM + component;\n" +
    "        rotated_qkv[output_index] = encoded;\n" +
    "        value_cache[cache_index] = encoded;\n" +
    "      }\n" +
    "    }\n" +
    (if physical.numerics.pair_retention() is Some(_) {
      ""
    } else {
      "    __syncwarp();\n"
    }) +
    "    }\n" +
    "  }\n",
  )
  match dispatch_canary {
    Some(value) =>
      out.write_string(
        "  if (threadIdx.x == 0 && head_begin == 0) atomicAdd(dispatch_canary, " +
        value.to_string() +
        "u);\n",
      )
    None => ()
  }
  out.write_string("}\n")
  if step_rotary_cache {
    out.write_string(
      qwen_step_rotary_preparation_source(symbol, physical.numerics, profile),
    )
  }
  @utf8.encode(out.to_string())
}
