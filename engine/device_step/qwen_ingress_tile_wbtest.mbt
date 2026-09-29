///|
test "full ingress admission consumes compiler row ownership instead of a fixed microtile" {
  let plan = qwen_fused_test_plan()
  let base = qwen_fused_test_blueprint(plan)
  let blueprint = {
    ..base,
    profile: @launch_contract.PagedKernelProfile::new(
      id=base.profile().id(),
      max_query_rows=base.profile().max_query_rows(),
      max_query_tokens=129,
      max_page_table_entries=base.profile().max_page_table_entries(),
    ),
  }
  for rows in [16, 32, 64] {
    let runtime = qwen_fused_test_runtime(
      plan,
      blueprint,
      full_ingress=true,
      ingress_query_tile_rows=rows,
    )
    let spans = admit_reusable_qwen_fused_attention_spans(
      plan,
      @worker_protocol.ModelPlanGeneration::first(),
      blueprint,
      runtime,
    )
    assert_eq(spans.length(), 28)
    assert_eq(spans[0].postprocess_launch().grid_x(), 128 / rows + 1)
    assert_true(spans[0].ingress_includes_projection())
    let rule = qwen_ingress_bucket_launch(spans[0])
    for tokens in [1, 16, 17, 32, 33, 64, 65, 128, 129] {
      assert_eq(
        paged_bucket_launch(rule, tokens).grid_x(),
        (tokens - 1) / rows + 1,
      )
    }
  }
  let wrong = qwen_fused_test_runtime(
    plan,
    blueprint,
    full_ingress=true,
    ingress_query_tile_rows=64,
    ingress_grid_x=9,
  )
  try
    admit_reusable_qwen_fused_attention_spans(
      plan,
      @worker_protocol.ModelPlanGeneration::first(),
      blueprint,
      wrong,
    )
  catch {
    MismatchedFusedGroupedOperation(Launch) => ()
    _ => fail("wrong geometry rejection")
  } noraise {
    _ => fail("legacy 16-row grid accepted for a 64-row module")
  }
  let legacy = qwen_fused_test_runtime(plan, blueprint, full_ingress=true)
  let spans = admit_reusable_qwen_fused_attention_spans(
    plan,
    @worker_protocol.ModelPlanGeneration::first(),
    blueprint,
    legacy,
  )
  assert_eq(spans[0].postprocess_launch().grid_x(), 9)
}
