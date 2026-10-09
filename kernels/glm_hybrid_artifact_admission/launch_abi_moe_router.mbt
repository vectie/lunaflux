///|
fn exact_moe_router_projection_abi(
  geometry : @glm_hybrid_kernel_requirements.GlmHybridKernelGeometry,
  hidden_width : Int,
  expert_count : Int,
  float32_router : Bool,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  if !float32_router ||
    hidden_width != geometry.hidden_size() ||
    expert_count <= 0 {
    return false
  }
  guard claim.operands is [counts, input, weight, logits] else { return false }
  guard checked_positive_mul(hidden_width.to_int64(), 4L)
    is Some(input_row_bytes) &&
    exact_rows(input.byte_count, input_row_bytes) is Some(maximum_rows) &&
    checked_positive_mul(hidden_width.to_int64(), expert_count.to_int64())
    is Some(weight_cells) &&
    checked_positive_mul(maximum_rows, expert_count.to_int64() * 4L)
    is Some(output_bytes) else {
    return false
  }
  maximum_rows <= geometry.context_length().to_int64() &&
  claim.function_symbol == "lunaflux_glm_hybrid_moe_router_f32_bf16_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(input, MoeRouterFloat32Input, input.byte_count, 16L) &&
  operand_is(weight, MoeRouterBf16Weights, weight_cells * 2L, 16L) &&
  operand_is(logits, MoeRouterFloat32LogitsOutput, output_bytes, 16L) &&
  claim.launch.grid_x().to_int64() ==
  (maximum_rows * expert_count.to_int64() + 255L) / 256L &&
  claim.launch.grid_y() == 1 &&
  claim.launch.grid_z() == 1 &&
  claim.launch.block_x() == 256 &&
  claim.launch.block_y() == 1 &&
  claim.launch.block_z() == 1 &&
  claim.launch.shared_memory_bytes() == 0
}
