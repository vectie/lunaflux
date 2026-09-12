///|
test "grouped postprocess launch reaches the complete reusable execution graph" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  for groups in [8, 32] {
    let runtime = qwen_fused_test_runtime(
      plan,
      blueprint,
      ingress_grid_y=groups,
    )
    let spans = admit_reusable_qwen_fused_attention_spans(
      plan,
      @worker_protocol.ModelPlanGeneration::first(),
      blueprint,
      runtime,
    )
    assert_eq(spans.length(), 28)
    assert_false(spans[0].ingress_includes_projection())
    assert_eq(spans[27].operation_indices(), (272, 273, 274, 275))
  }
}

///|
test "postprocess grouping cannot change the full projection ingress ABI" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  for pair in [(true, 8), (false, 7), (false, 9), (false, 16)] {
    let (full_ingress, groups) = pair
    let runtime = qwen_fused_test_runtime(
      plan,
      blueprint,
      full_ingress~,
      ingress_grid_y=groups,
    )
    try
      ignore(
        admit_reusable_qwen_fused_attention_spans(
          plan,
          @worker_protocol.ModelPlanGeneration::first(),
          blueprint,
          runtime,
        ),
      )
    catch {
      MismatchedFusedGroupedOperation(Launch) => ()
      _ => fail("unexpected launch rejection")
    } noraise {
      _ => fail("incompatible head layout reached the execution graph")
    }
  }
}
