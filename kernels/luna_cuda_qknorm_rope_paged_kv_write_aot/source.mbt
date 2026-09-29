///|
let postprocess_production_symbol : String = "lunaflux_qwen_qknorm_rope_kvwrite_bf16_head128_production_v2"

///|
fn postprocess_cuda_float_literal(value : Double) -> String {
  let rendered = value.to_string()
  if rendered.contains(".") || rendered.contains("e") || rendered.contains("E") {
    rendered + "f"
  } else {
    rendered + ".0f"
  }
}

///|
fn postprocess_source(
  semantic : @luna_fusion_plan.QkvQkNormPositionedRopePagedKvWritePlan,
  layout : @device_layout.DeviceKvLayout,
  profile : @launch_contract.PagedKernelProfile,
  grid_x : Int,
  head_map : @luna_projection_tile_compiler.AttentionIngressHeadMap,
  numerics : @luna_projection_tile_compiler.AttentionIngressNumerics,
  physical : @elementwise_physical_ir.IngressProgram,
) -> Bytes {
  let (_, output_width) = semantic.projection_shape()
  let (query_heads, kv_heads, _) = semantic.attention_shape()
  let (packed_heads, qk_heads, kv_width) = physical.geometry()
  let (head_groups, heads_per_workgroup, _) = head_map.geometry()
  let (head_dimension, lanes_per_head, _) = physical.ownership().geometry()
  let numerics_source = @luna_cuda_attention_ingress_source.lower_attention_ingress_numerics(
    numerics,
  )
  @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n" +
    numerics_source +
    "#define LF_BLOCK_THREADS \{heads_per_workgroup * lanes_per_head}\n" +
    "#define LF_HEADS_PER_GROUP \{heads_per_workgroup}\n" +
    "#define LF_LANES_PER_HEAD \{lanes_per_head}\n" +
    "#define LF_OUTPUT_WIDTH " +
    output_width.to_string() +
    "\n" +
    "#define LF_QUERY_HEADS " +
    query_heads.to_string() +
    "\n" +
    "#define LF_KV_HEADS " +
    kv_heads.to_string() +
    "\n" +
    "#define LF_QK_HEADS " +
    qk_heads.to_string() +
    "\n" +
    "#define LF_PACKED_HEADS " +
    packed_heads.to_string() +
    "\n" +
    "#define LF_HEAD_DIM " +
    head_dimension.to_string() +
    "\n" +
    "#define LF_KV_WIDTH " +
    kv_width.to_string() +
    "\n" +
    "#define LF_TOKENS_PER_PAGE " +
    layout.tokens_per_page().to_string() +
    "\n" +
    "#define LF_TOTAL_PAGE_COUNT " +
    layout.total_page_count().to_string() +
    "\n" +
    "#define LF_PAGE_STRIDE_BF16 " +
    (layout.page_stride_bytes() / 2L).to_string() +
    "ULL\n" +
    "#define LF_NORM_EPSILON " +
    postprocess_cuda_float_literal(semantic.norm_epsilon()) +
    "\n" +
    "#define LF_ROPE_THETA " +
    postprocess_cuda_float_literal(semantic.rope_theta()) +
    "\n" +
    "extern \"C\" __global__ void " +
    postprocess_production_symbol +
    "(\n" +
    "  const int32_t* __restrict__ step_counts,\n" +
    "  const int32_t* __restrict__ query_positions,\n" +
    "  const int32_t* __restrict__ query_row_offsets,\n" +
    "  const int32_t* __restrict__ sequence_lengths,\n" +
    "  const int32_t* __restrict__ page_table_offsets,\n" +
    "  const int32_t* __restrict__ page_table_page_indices,\n" +
    "  const __nv_bfloat16* __restrict__ projected_qkv,\n" +
    "  const __nv_bfloat16* __restrict__ query_norm_weight,\n" +
    "  const __nv_bfloat16* __restrict__ key_norm_weight,\n" +
    "  __nv_bfloat16* __restrict__ rotated_qkv,\n" +
    "  __nv_bfloat16* __restrict__ key_cache,\n" +
    "  __nv_bfloat16* __restrict__ value_cache) {\n" +
    "  if (blockDim.x != LF_BLOCK_THREADS || blockIdx.z != 0 || gridDim.x == 0 || gridDim.x > " +
    grid_x.to_string() +
    " || gridDim.y != \{head_groups}) return;\n" +
    "  const int lane = (int)threadIdx.x % LF_LANES_PER_HEAD;\n" +
    "  const int subgroup = (int)threadIdx.x / LF_LANES_PER_HEAD;\n" +
    "  const int head = (int)blockIdx.y * LF_HEADS_PER_GROUP + subgroup;\n" +
    "  if (head >= LF_PACKED_HEADS) return;\n" +
    "  const int prefill = step_counts[0], decode = step_counts[1];\n" +
    "  const int rows = step_counts[2], tokens = step_counts[3];\n" +
    "  const int table_cells = step_counts[4];\n" +
    "  if (prefill < 0 || decode < 0 || rows <= 0 || rows > " +
    profile.max_query_rows().to_string() +
    " || decode != rows - prefill || tokens <= 0 || tokens > " +
    profile.max_query_tokens().to_string() +
    " || table_cells <= 0 || table_cells > " +
    profile.max_page_table_entries().to_string() +
    " ||\n" +
    "      query_row_offsets[0] != 0 || query_row_offsets[rows] != tokens ||\n" +
    "      page_table_offsets[0] != 0 || page_table_offsets[rows] != table_cells) return;\n" +
    "  __shared__ float head_storage[LF_HEADS_PER_GROUP * LF_HEAD_DIM];\n" +
    "  float* head_values = head_storage + subgroup * LF_HEAD_DIM;\n" +
    "  for (int token = (int)blockIdx.x; token < tokens; token += (int)gridDim.x) {\n" +
    "    int row = 0;\n" +
    "    while (row + 1 < rows && token >= query_row_offsets[row + 1]) ++row;\n" +
    "    const int position = query_positions[token];\n" +
    "    if (query_row_offsets[row] < 0 || query_row_offsets[row] >= query_row_offsets[row + 1] ||\n" +
    "        query_row_offsets[row + 1] > tokens || token < query_row_offsets[row] ||\n" +
    "        token >= query_row_offsets[row + 1] || sequence_lengths[row] <= 0 ||\n" +
    "        position < 0 || position >= sequence_lengths[row]) return;\n" +
    "    const int logical_page = position / LF_TOKENS_PER_PAGE;\n" +
    "    const int page_begin = page_table_offsets[row];\n" +
    "    const int page_end = page_table_offsets[row + 1];\n" +
    "    if (page_begin < 0 || page_begin >= page_end || page_end > table_cells ||\n" +
    "        logical_page < 0 || logical_page >= page_end - page_begin) return;\n" +
    "    const int physical_page = page_table_page_indices[page_begin + logical_page];\n" +
    "    if (physical_page < 0 || physical_page >= LF_TOTAL_PAGE_COUNT) return;\n" +
    "    const unsigned long long head_base =\n" +
    "      ((unsigned long long)token * LF_PACKED_HEADS + (unsigned long long)head) * LF_HEAD_DIM;\n" +
    postprocess_effect_source(physical) +
    "  }\n" +
    "}\n",
  )
}
