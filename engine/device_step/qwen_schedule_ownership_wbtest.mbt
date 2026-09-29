///|
test "packed ingress preserves head ownership through graph bucket launch" {
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
  for heads in [2, 4] {
    let runtime = qwen_fused_test_runtime(
      plan,
      blueprint,
      full_ingress=true,
      ingress_query_tile_rows=32,
      ingress_heads_per_cta=heads,
      ingress_grid_y=32 / heads,
    )
    let spans = admit_reusable_qwen_fused_attention_spans(
      plan,
      @worker_protocol.ModelPlanGeneration::first(),
      blueprint,
      runtime,
    )
    let rule = qwen_ingress_bucket_launch(spans[0])
    for tokens in [1, 31, 32, 33, 64] {
      let launch = paged_bucket_launch(rule, tokens)
      assert_eq(launch.grid_x(), (tokens - 1) / 32 + 1)
      assert_eq(launch.grid_y(), 32 / heads)
    }
    let wrong = qwen_fused_test_runtime(
      plan,
      blueprint,
      full_ingress=true,
      ingress_query_tile_rows=32,
      ingress_heads_per_cta=heads,
      ingress_grid_y=32,
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
      _ => fail("wrong head ownership error")
    } noraise {
      _ => fail("one-head grid accepted for packed ownership")
    }
  }
}

///|
test "blockwise decode has a distinct numerical admission and no implicit split ABI" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  let runtime = qwen_fused_test_runtime(
    plan,
    blueprint,
    phase_specific_attention=true,
    blockwise_decode=true,
  )
  assert_true(runtime.qwen_paged_attention_decode_split_module() is None)
  let spans = admit_reusable_qwen_fused_attention_spans(
    plan,
    @worker_protocol.ModelPlanGeneration::first(),
    blueprint,
    runtime,
  )
  assert_eq(spans.length(), 28)
  assert_eq(
    spans[0].decode_function_symbol(),
    "lunaflux_attention_decode_tile_compiler_v1_blockwise_f32_v1",
  )
  for blockwise in [false, true] {
    let wrong = qwen_fused_test_runtime(
      plan,
      blueprint,
      phase_specific_attention=true,
      blockwise_decode=blockwise,
      mismatched_decode_law=true,
    )
    try
      admit_reusable_qwen_fused_attention_spans(
        plan,
        @worker_protocol.ModelPlanGeneration::first(),
        blueprint,
        wrong,
      )
    catch {
      MismatchedFusedGroupedOperation(KernelAbi) => ()
      _ => fail("wrong numerical law error")
    } noraise {
      _ => fail("mismatched decode numerical law accepted")
    }
  }
}
