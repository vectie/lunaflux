///|
fn artifact_test_moe_router_abi() -> (
  String,
  @launch_contract.AotLaunchDimensions,
  Array[GlmHybridLaunchOperand],
) {
  (
    "lunaflux_glm_hybrid_moe_router_f32_bf16_v1",
    artifact_test_exact_launch(1, 0),
    [
      artifact_test_operand(StepCountsInput, 20L, 4L),
      artifact_test_operand(MoeRouterFloat32Input, 64L, 16L),
      artifact_test_operand(MoeRouterBf16Weights, 128L, 16L),
      artifact_test_operand(MoeRouterFloat32LogitsOutput, 64L, 16L),
    ],
  )
}
