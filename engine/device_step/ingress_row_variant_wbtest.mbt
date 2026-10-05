///|
test "small ingress keeps its own tile geometry and does not replace long prefill" {
  let plan = qwen_fused_test_plan()
  let base = qwen_fused_test_blueprint(plan)
  let blueprint = {
    ..base,
    profile: @launch_contract.PagedKernelProfile::new(
      id=base.profile().id(),
      max_query_rows=3,
      max_query_tokens=129,
      max_page_table_entries=base.profile().max_page_table_entries(),
    ),
  }
  let variant = @luna_fused_artifact_admission.ReusableFusedRuntimeRowVariant::new(
    maximum_token_rows=64,
    query_tile_rows=16,
    function_symbol="lunaflux_fused_qwen_qkv_qknorm_rope_kvwrite_bf16_head128_production_cached_rotary_v3",
    launch=@launch_contract.AotLaunchDimensions::new(
      grid_x=4,
      grid_y=32,
      grid_z=1,
      block_x=128,
      block_y=1,
      block_z=1,
      shared_memory_bytes=8192,
    ),
    module_bytes=b"small-ingress",
  )
  let runtime = qwen_fused_test_runtime(
    plan,
    blueprint,
    full_ingress=true,
    step_rotary_cache=true,
    ingress_query_tile_rows=64,
    ingress_row_variant=variant,
  )
  let spans = admit_reusable_qwen_fused_attention_spans(
    plan,
    @worker_protocol.ModelPlanGeneration::first(),
    blueprint,
    runtime,
  )
  assert_eq(spans.length(), 28)
  assert_true(spans[0].ingress_row_variant == Some(variant))
  let small : PagedProjectionVariant = {
    base_function_index: 2,
    function_index: 9,
    maximum_token_rows: 64,
    launch: QueryTileCappedGridX(
      variant.launch_dimensions(),
      variant.query_tile_rows(),
    ),
  }
  for bound in [2, 8, 16, 17, 32, 64] {
    let selected = select_projection_variant([small], 2, bound).unwrap()
    assert_eq(selected.function_index, 9)
    let launch = paged_bucket_launch(selected.launch, bound)
    assert_eq(launch.grid_x(), (bound - 1) / 16 + 1)
    assert_eq(launch.shared_memory_bytes(), 8192)
  }
  for bound in [0, 1, 65, 128, 129] {
    assert_true(select_projection_variant([small], 2, bound) is None)
  }
  // Retained baseline has a 64-row domain, even when the small entry is present.
  assert_eq(
    paged_bucket_launch(qwen_ingress_bucket_launch(spans[0]), 129).grid_x(),
    3,
  )
}
