///|
test "dual score decode reaches every layer with its declared law" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  for wrong in [false, true] {
    let runtime = qwen_fused_test_runtime(
      plan,
      blueprint,
      phase_specific_attention=true,
      dual_score_decode=true,
      mismatched_decode_law=wrong,
    )
    try {
      let spans = admit_reusable_qwen_fused_attention_spans(
        plan,
        @worker_protocol.ModelPlanGeneration::first(),
        blueprint,
        runtime,
      )
      if wrong {
        fail("dual score runtime relabeled as strict decode")
      }
      assert_eq(spans.length(), 28)
      assert_eq(
        spans[0].decode_function_symbol(),
        "lunaflux_attention_decode_tile_compiler_v1_dual_score_blockwise_f32_v2",
      )
    } catch {
      MismatchedFusedGroupedOperation(KernelAbi) if wrong => ()
      _ => fail("unexpected dual score span error")
    }
  }
}
