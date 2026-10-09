///|
fn exact_moe_sigmoid_correction_abi(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  expert_count : Int,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  if expert_count <= 0 {
    return false
  }
  guard claim.operands is [counts, logits, bias, scores, choice_scores] else {
    return false
  }
  let row_bytes = expert_count.to_int64() * 4L
  guard exact_rows(logits.byte_count, row_bytes) is Some(maximum_rows) &&
    checked_positive_mul(maximum_rows, row_bytes) is Some(score_bytes) else {
    return false
  }
  maximum_rows <= requirements.geometry().context_length().to_int64() &&
  claim.function_symbol == "lunaflux_glm_hybrid_moe_sigmoid_correction_f32_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(logits, MoeRouterFloat32LogitsOutput, score_bytes, 16L) &&
  operand_is(bias, MoeCorrectionBiasInput, row_bytes, 16L) &&
  operand_is(scores, MoeUncorrectedScoresOutput, score_bytes, 16L) &&
  operand_is(choice_scores, MoeChoiceScoresOutput, score_bytes, 16L) &&
  scores.role != choice_scores.role &&
  exact_launch(claim.launch, maximum_rows * expert_count.to_int64(), 0)
}
