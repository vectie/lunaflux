///|
fn exact_moe_grouped_routing_abi(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  routed_expert_count : Int,
  group_count : Int,
  group_score_topk : Int,
  selected_group_count : Int,
  experts_per_token : Int,
  normalize : Bool,
  scaling : Double,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  if routed_expert_count <= 0 ||
    group_count <= 0 ||
    routed_expert_count % group_count != 0 ||
    group_score_topk <= 0 ||
    group_score_topk > routed_expert_count / group_count ||
    selected_group_count <= 0 ||
    selected_group_count > group_count ||
    selected_group_count * (routed_expert_count / group_count) <
    experts_per_token ||
    experts_per_token <= 0 ||
    experts_per_token > routed_expert_count ||
    scaling.is_nan() ||
    scaling.is_inf() ||
    scaling <= 0.0 {
    return false
  }
  guard claim.operands is [counts, scores, choice_scores, indices, weights] else {
    return false
  }
  let score_row_bytes = routed_expert_count.to_int64() * 4L
  let output_row_bytes = experts_per_token.to_int64() * 4L
  guard exact_rows(scores.byte_count, score_row_bytes) is Some(maximum_rows) &&
    checked_positive_mul(maximum_rows, score_row_bytes) is Some(score_bytes) &&
    checked_positive_mul(maximum_rows, output_row_bytes) is Some(output_bytes) else {
    return false
  }
  ignore(normalize)
  maximum_rows <= requirements.geometry().context_length().to_int64() &&
  claim.function_symbol == "lunaflux_glm_hybrid_moe_grouped_routing_f32_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(scores, MoeUncorrectedScoresOutput, score_bytes, 16L) &&
  operand_is(choice_scores, MoeChoiceScoresOutput, score_bytes, 16L) &&
  operand_is(indices, MoeSelectedIndicesOutput, output_bytes, 4L) &&
  operand_is(weights, MoeSelectedWeightsOutput, output_bytes, 16L) &&
  claim.launch.grid_x().to_int64() == maximum_rows &&
  claim.launch.grid_y() == 1 &&
  claim.launch.grid_z() == 1 &&
  claim.launch.block_x() == 1 &&
  claim.launch.block_y() == 1 &&
  claim.launch.block_z() == 1 &&
  claim.launch.shared_memory_bytes() == 0
}
