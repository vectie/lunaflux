///|
fn exact_kda_projection_abi(
  geometry : @glm_hybrid_kernel_requirements.GlmHybridKernelGeometry,
  head_count : Int,
  head_dimension : Int,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  let hidden_width = geometry.hidden_size()
  let control_rank = head_dimension
  if hidden_width != 4096 || head_count != 64 || head_dimension != 128 {
    return false
  }
  guard claim.operands
    is [
      counts,
      hidden,
      q_weight,
      k_weight,
      v_weight,
      forget_a_weight,
      forget_b_weight,
      input_gate_weight,
      query,
      key,
      value,
      forget_low_rank,
      raw_forget,
      beta,
    ] else {
    return false
  }
  guard checked_positive_mul(hidden_width.to_int64(), 2L)
    is Some(hidden_row_bytes) &&
    exact_rows(hidden.byte_count, hidden_row_bytes) is Some(maximum_rows) &&
    checked_positive_mul(head_count.to_int64(), head_dimension.to_int64())
    is Some(projected_width) &&
    checked_positive_mul(projected_width, 2L) is Some(projected_row_bytes) &&
    checked_positive_mul(maximum_rows, projected_row_bytes)
    is Some(projected_bytes) &&
    checked_positive_mul(maximum_rows, control_rank.to_int64() * 2L)
    is Some(low_bytes) &&
    checked_positive_mul(maximum_rows, head_count.to_int64() * 2L)
    is Some(beta_bytes) &&
    checked_positive_mul(projected_width, hidden_width.to_int64())
    is Some(qkv_weight_cells) &&
    checked_positive_mul(qkv_weight_cells, 2L) is Some(qkv_weight_bytes) &&
    checked_positive_mul(control_rank.to_int64(), hidden_width.to_int64())
    is Some(forget_a_cells) &&
    checked_positive_mul(forget_a_cells, 2L) is Some(forget_a_bytes) &&
    checked_positive_mul(projected_width, control_rank.to_int64())
    is Some(forget_b_cells) &&
    checked_positive_mul(forget_b_cells, 2L) is Some(forget_b_bytes) &&
    checked_positive_mul(head_count.to_int64(), hidden_width.to_int64())
    is Some(gate_weight_cells) &&
    checked_positive_mul(gate_weight_cells, 2L) is Some(gate_weight_bytes) else {
    return false
  }
  maximum_rows <= geometry.context_length().to_int64() &&
  claim.function_symbol == "lunaflux_glm_hybrid_kda_projection_bf16_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(hidden, HiddenInput, hidden.byte_count, 16L) &&
  operand_is(q_weight, KdaQueryProjectionWeights, qkv_weight_bytes, 16L) &&
  operand_is(k_weight, KdaKeyProjectionWeights, qkv_weight_bytes, 16L) &&
  operand_is(v_weight, KdaValueProjectionWeights, qkv_weight_bytes, 16L) &&
  operand_is(forget_a_weight, KdaForgetLowRankWeights, forget_a_bytes, 16L) &&
  operand_is(forget_b_weight, KdaForgetExpansionWeights, forget_b_bytes, 16L) &&
  operand_is(input_gate_weight, KdaInputGateWeights, gate_weight_bytes, 16L) &&
  operand_is(query, KdaProjectedQueryOutput, projected_bytes, 16L) &&
  operand_is(key, KdaProjectedKeyOutput, projected_bytes, 16L) &&
  operand_is(value, KdaProjectedValueOutput, projected_bytes, 16L) &&
  operand_is(forget_low_rank, KdaForgetLowRankOutput, low_bytes, 16L) &&
  operand_is(raw_forget, KdaRawForgetOutput, projected_bytes, 16L) &&
  operand_is(beta, KdaBetaOutput, beta_bytes, 16L) &&
  claim.launch.grid_x() == 1 &&
  claim.launch.grid_y() == 1 &&
  claim.launch.grid_z() == 1 &&
  claim.launch.block_x() == 1 &&
  claim.launch.block_y() == 1 &&
  claim.launch.block_z() == 1 &&
  claim.launch.shared_memory_bytes() == 0
}
