///|
fn artifact_test_moe_score_abi() -> (
  String,
  @launch_contract.AotLaunchDimensions,
  Array[GlmHybridLaunchOperand],
) {
  (
    "lunaflux_glm_hybrid_moe_sigmoid_correction_f32_v1",
    artifact_test_exact_launch(1, 0),
    [
      artifact_test_operand(StepCountsInput, 20L, 4L),
      artifact_test_operand(MoeRouterFloat32LogitsOutput, 64L, 16L),
      artifact_test_operand(MoeCorrectionBiasInput, 32L, 16L),
      artifact_test_operand(MoeUncorrectedScoresOutput, 64L, 16L),
      artifact_test_operand(MoeChoiceScoresOutput, 64L, 16L),
    ],
  )
}
