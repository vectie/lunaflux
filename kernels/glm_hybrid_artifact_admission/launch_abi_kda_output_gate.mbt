///|
fn exact_kda_output_gate_projection_abi(
  geometry : @glm_hybrid_kernel_requirements.GlmHybridKernelGeometry,
  hidden_width : Int,
  head_count : Int,
  head_dimension : Int,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  if hidden_width != geometry.hidden_size() ||
    hidden_width != 4096 ||
    head_count != 64 ||
    head_dimension != 128 {
    return false
  }
  guard claim.operands
    is [counts, hidden, first_weight, second_weight, low, gate] else {
    return false
  }
  guard checked_positive_mul(hidden_width.to_int64(), 2L)
    is Some(hidden_row_bytes) &&
    exact_rows(hidden.byte_count, hidden_row_bytes) is Some(maximum_rows) &&
    checked_positive_mul(head_count.to_int64(), head_dimension.to_int64())
    is Some(output_width) &&
    checked_positive_mul(maximum_rows, head_dimension.to_int64() * 2L)
    is Some(low_bytes) &&
    checked_positive_mul(maximum_rows, output_width * 2L) is Some(gate_bytes) &&
    checked_positive_mul(head_dimension.to_int64(), hidden_width.to_int64())
    is Some(first_cells) &&
    checked_positive_mul(first_cells, 2L) is Some(first_bytes) &&
    checked_positive_mul(output_width, head_dimension.to_int64())
    is Some(second_cells) &&
    checked_positive_mul(second_cells, 2L) is Some(second_bytes) else {
    return false
  }
  maximum_rows <= geometry.context_length().to_int64() &&
  claim.function_symbol == "lunaflux_glm_hybrid_kda_output_gate_bf16_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(hidden, HiddenInput, hidden.byte_count, 16L) &&
  operand_is(first_weight, KdaOutputGateLowRankWeights, first_bytes, 16L) &&
  operand_is(second_weight, KdaOutputGateExpansionWeights, second_bytes, 16L) &&
  operand_is(low, KdaOutputGateLowRankOutput, low_bytes, 16L) &&
  operand_is(gate, KdaOutputGateOutput, gate_bytes, 16L) &&
  claim.launch.grid_x() == 1 &&
  claim.launch.grid_y() == 1 &&
  claim.launch.grid_z() == 1 &&
  claim.launch.block_x() == 1 &&
  claim.launch.block_y() == 1 &&
  claim.launch.block_z() == 1 &&
  claim.launch.shared_memory_bytes() == 0
}
