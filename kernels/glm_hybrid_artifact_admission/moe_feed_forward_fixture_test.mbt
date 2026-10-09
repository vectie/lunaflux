///|
fn artifact_test_moe_routed_expert_abi() -> (
  String,
  @launch_contract.AotLaunchDimensions,
  Array[GlmHybridLaunchOperand],
) {
  (
    "lunaflux_glm_hybrid_moe_routed_experts_bf16_v1",
    artifact_test_exact_launch(1, 0),
    [
      artifact_test_operand(StepCountsInput, 20L, 4L),
      artifact_test_operand(MoeFeedForwardInput, 32L, 16L),
      artifact_test_operand(MoeSelectedIndicesOutput, 16L, 4L),
      artifact_test_operand(MoeSelectedWeightsOutput, 16L, 16L),
      artifact_test_operand(MoeExpertGateWeights, 2048L, 16L),
      artifact_test_operand(MoeExpertUpWeights, 2048L, 16L),
      artifact_test_operand(MoeExpertDownWeights, 2048L, 16L),
      artifact_test_operand(MoeRoutedOutput, 32L, 16L),
    ],
  )
}

///|
fn artifact_test_moe_shared_expert_abi() -> (
  String,
  @launch_contract.AotLaunchDimensions,
  Array[GlmHybridLaunchOperand],
) {
  (
    "lunaflux_glm_hybrid_moe_shared_expert_bf16_v1",
    artifact_test_exact_launch(1, 0),
    [
      artifact_test_operand(StepCountsInput, 20L, 4L),
      artifact_test_operand(MoeFeedForwardInput, 32L, 16L),
      artifact_test_operand(MoeSharedGateWeights, 256L, 16L),
      artifact_test_operand(MoeSharedUpWeights, 256L, 16L),
      artifact_test_operand(MoeSharedDownWeights, 256L, 16L),
      artifact_test_operand(MoeSharedOutput, 32L, 16L),
    ],
  )
}

///|
fn artifact_test_moe_combine_abi() -> (
  String,
  @launch_contract.AotLaunchDimensions,
  Array[GlmHybridLaunchOperand],
) {
  (
    "lunaflux_glm_hybrid_moe_routed_shared_add_bf16_v1",
    artifact_test_exact_launch(1, 0),
    [
      artifact_test_operand(StepCountsInput, 20L, 4L),
      artifact_test_operand(MoeRoutedOutput, 32L, 16L),
      artifact_test_operand(MoeSharedOutput, 32L, 16L),
      artifact_test_operand(MoeCombinedOutput, 32L, 16L),
    ],
  )
}
