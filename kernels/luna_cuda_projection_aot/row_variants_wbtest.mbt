///|
test "offline small-row winner survives maximum-profile AOT lowering" {
  let parameters : ProjectionParameters = {
    family: DenseProjection,
    input_width: 1024,
    output_width: 1024,
    query_width: 0,
    intermediate_width: 0,
  }
  let target = projection_fixture_target()
  let baseline = projection_build_runtime_dispatch(parameters, 256, target, [])
  let candidates = baseline
    .dispatch_plan(8)
    .prefill()
    .output_distribution_candidates()
  let mut selected : @luna_projection_strategy.ProjectionStrategy? = None
  for candidate in candidates {
    if candidate.output_tiles_per_workgroup() == 2 {
      selected = Some(candidate)
    }
  }
  let small = selected.unwrap()
  let record = @luna_projection_strategy.ProjectionAutotuneRecord::new(
    key=@luna_projection_strategy.ProjectionAutotuneKey::new(
      backend=projection_cuda_backend_id(),
      target=projection_cuda_target_id(target),
      numeric_type=ProjectionBFloat16,
      shape=projection_strategy_shape(parameters),
      strategy=small,
    ),
    median_latency_ns=1L,
    sample_count=3,
  )
  let runtime = projection_build_runtime_dispatch(parameters, 256, target, [
    record,
  ])
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=32,
    max_query_tokens=256,
    max_page_table_entries=8192,
  )
  let (source, variants) = projection_row_variant_sources(
    parameters,
    profile,
    runtime,
    runtime.dispatch_plan(256),
    "projection",
  )
  assert_eq(variants.length(), 1)
  assert_eq(variants[0].maximum_token_rows(), 8)
  assert_eq(variants[0].function_symbol(), "projection_rows8")
  assert_eq(
    variants[0].dimensions().block_x(),
    small.output_tiles_per_workgroup() * 32,
  )
  assert_true(source.contains("token_count > 8"))
  assert_true(source.contains("void projection_rows8("))
  let (fallback_source, fallback_variants) = projection_row_variant_sources(
    parameters,
    profile,
    baseline,
    baseline.dispatch_plan(256),
    "projection",
  )
  assert_eq(fallback_source, "")
  assert_true(fallback_variants.is_empty())
  if @env.get_env_var("LUNA_TEST_EXPORT_ROW_VARIANTS") == Some("1") {
    let dispatch = runtime.dispatch_plan(256)
    let compilation = @luna_projection_tile_compiler.compile_projection_tile(
      @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
        dispatch~,
        maximum_rows=32,
        maximum_tokens=256,
        demand=AllTokenRows,
        single_row_numerics=AllowDeterministicDotTree,
      ),
    )
    println("LUNA_ROW_VARIANTS_BEGIN")
    println(
      @utf8.decode(
        projection_source(
          "projection", parameters, 256, 32, dispatch, compilation, false, true,
        ),
      ),
    )
    println(source)
    println("LUNA_ROW_VARIANTS_END")
  }
}
