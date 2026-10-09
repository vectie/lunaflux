///|
fn artifact_test_moe_grouped_routing_abi() -> (
  String,
  @launch_contract.AotLaunchDimensions,
  Array[GlmHybridLaunchOperand],
) {
  (
    "lunaflux_glm_hybrid_moe_grouped_routing_f32_v1",
    try! @launch_contract.AotLaunchDimensions::new(
      grid_x=2,
      grid_y=1,
      grid_z=1,
      block_x=1,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    [
      artifact_test_operand(StepCountsInput, 20L, 4L),
      artifact_test_operand(MoeUncorrectedScoresOutput, 64L, 16L),
      artifact_test_operand(MoeChoiceScoresOutput, 64L, 16L),
      artifact_test_operand(MoeSelectedIndicesOutput, 16L, 4L),
      artifact_test_operand(MoeSelectedWeightsOutput, 16L, 16L),
    ],
  )
}
