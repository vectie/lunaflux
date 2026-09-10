///|
// A synthetic record drives selection only; no test latency is installed in a
// production policy or reclassified as a measurement for a larger bucket.
fn row_variant_test_record(
  parameters : ProjectionParameters,
  strategy : @luna_projection_strategy.ProjectionStrategy,
) -> @luna_projection_strategy.ProjectionAutotuneRecord raise {
  @luna_projection_strategy.ProjectionAutotuneRecord::new(
    key=@luna_projection_strategy.ProjectionAutotuneKey::new(
      backend=projection_cuda_backend_id(),
      target=projection_cuda_target_id(projection_fixture_target()),
      numeric_type=ProjectionBFloat16,
      shape=projection_strategy_shape(parameters),
      strategy~,
    ),
    median_latency_ns=1L,
    sample_count=3,
  )
}

///|
fn row_variant_test_profile() -> @launch_contract.PagedKernelProfile raise {
  @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=32,
    max_query_tokens=1024,
    max_page_table_entries=8192,
  )
}

///|
fn row_variant_primary_test_source(
  parameters : ProjectionParameters,
  dispatch : @luna_projection_strategy.ProjectionDispatchPlan,
  maximum_tokens? : Int = 1024,
) -> String raise {
  let compilation = @luna_projection_tile_compiler.compile_projection_tile(
    @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
      dispatch~,
      maximum_rows=32,
      maximum_tokens~,
      demand=if parameters.family == LanguageModelHead {
        QueryRowEnds
      } else {
        AllTokenRows
      },
      single_row_numerics=AllowDeterministicDotTree,
    ),
  )
  @utf8.decode(
    projection_source(
      "baseline",
      parameters,
      maximum_tokens,
      32,
      dispatch,
      compilation,
      false,
      true,
      max_registers=128,
    ),
  )
}

///|
test "head residency-only winner remains executable inside a larger profile" {
  let parameters : ProjectionParameters = {
    family: LanguageModelHead,
    input_width: 1024,
    output_width: 151936,
    query_width: 0,
    intermediate_width: 0,
  }
  let target = projection_fixture_target()
  let baseline = projection_build_runtime_dispatch(parameters, 1024, target, [])
  let resident = baseline
    .dispatch_plan(256)
    .prefill()
    .with_selected_row_residency(stable_id=3008)
  let runtime = projection_build_runtime_dispatch(parameters, 1024, target, [
    row_variant_test_record(parameters, resident),
  ])
  let primary = runtime.dispatch_plan(1024)
  assert_eq(runtime.select(256).source(), OfflineAutotune)
  assert_eq(runtime.select(257).source(), StaticFallback)
  let profile = row_variant_test_profile()
  let (source, variants) = projection_row_variant_sources(
    parameters,
    profile,
    runtime,
    primary,
    "head",
    max_registers=128,
  )
  assert_eq(variants.length(), 1)
  let variant = variants[0]
  assert_eq(variant.maximum_token_rows(), 256)
  assert_eq(variant.function_symbol(), "head_rows256")
  assert_eq(variant.dimensions().shared_memory_bytes(), 40960)
  assert_eq(
    projection_dimensions(profile, parameters, primary).shared_memory_bytes(),
    8704,
  )
  assert_true(source.contains("void head_rows256("))
  assert_true(source.contains("const int32_t *row_offsets"))
  assert_true(source.contains("token_count > 256"))
  assert_true(source.contains("row_count > 32"))
  assert_true(source.contains("element < 16384"))
  assert_true(source.contains("inner < 1024; inner += 16"))
  assert_true(source.contains("operand_index(local_row, inner + local_k)"))
  assert_false(source.contains("wmma::"))
  assert_false(source.contains("sampling"))
  if @env.get_env_var("LUNA_TEST_EXPORT_ROW_VARIANT_FAMILIES") == Some("1") {
    println("LUNA_SOURCE_BEGIN head-resident-256-in-1024")
    println(row_variant_primary_test_source(parameters, primary))
    println(source)
    println("LUNA_SOURCE_END")
  }
}

///|
test "small-row MLP winner exports gate-up and down under one bounded ABI" {
  let parameters : ProjectionParameters = {
    family: GatedMlp,
    input_width: 1024,
    output_width: 1024,
    query_width: 0,
    intermediate_width: 3072,
  }
  let target = projection_fixture_target()
  let baseline = projection_build_runtime_dispatch(parameters, 1024, target, [])
  for groups in [1, 2, 4, 8] {
    let candidates = baseline
      .dispatch_plan(8)
      .prefill()
      .output_distribution_candidates()
    let selected = for candidate in candidates {
      if candidate.output_tiles_per_workgroup() == groups {
        break candidate
      }
    } nobreak {
      fail("missing legal output distribution")
    }
    let runtime = projection_build_runtime_dispatch(parameters, 1024, target, [
      row_variant_test_record(parameters, selected),
    ])
    let primary = runtime.dispatch_plan(1024)
    let profile = row_variant_test_profile()
    let (source, variants) = projection_row_variant_sources(
      parameters,
      profile,
      runtime,
      primary,
      "mlp",
      max_registers=128,
    )
    assert_eq(variants.length(), 1)
    let variant = variants[0]
    assert_eq(variant.maximum_token_rows(), 8)
    assert_eq(variant.dimensions().block_x(), groups * 32)
    assert_eq(variant.dimensions().shared_memory_bytes(), groups * 1024)
    assert_eq(variant.dimensions().grid_x(), 3072 / groups)
    assert_true(source.contains("void mlp_rows8("))
    assert_true(source.contains("void mlp_rows8_down("))
    assert_true(source.contains("__nv_bfloat16 *workspace"))
    assert_true(source.contains("token_count > 8"))
    assert_true(source.contains("blockDim.x != \{groups * 32}"))
    assert_true(source.contains("sibling_storage"))
    assert_eq(runtime.select(9).source(), StaticFallback)
    // The exported companion borrows the existing maximum-profile workspace;
    // changing work distribution cannot change its tensor storage contract.
    assert_eq(
      projection_workspace_bytes(parameters, profile, primary),
      projection_workspace_bytes(parameters, profile, runtime.dispatch_plan(8)),
    )
    if @env.get_env_var("LUNA_TEST_EXPORT_ROW_VARIANT_FAMILIES") == Some("1") {
      println("LUNA_SOURCE_BEGIN mlp-\{groups}-8-in-1024")
      println(row_variant_primary_test_source(parameters, primary))
      println(source)
      println("LUNA_SOURCE_END")
    }
  }
}

///|
test "monolithic scalar families cannot advertise matrix row variants" {
  for family in [GatedMlp, LanguageModelHead] {
    let parameters : ProjectionParameters = {
      family,
      input_width: 33,
      output_width: 33,
      query_width: 0,
      intermediate_width: if family == GatedMlp {
        33
      } else {
        0
      },
    }
    let target = projection_fixture_target()
    let baseline = projection_build_runtime_dispatch(parameters, 1024, target, [])
    let fallback = baseline.dispatch_plan(8).prefill()
    let scalar = @luna_projection_strategy.ProjectionStrategy::new(
      stable_id=4001,
      phase=Prefill,
      kernel_class=ScalarTile,
      row_bucket=8,
      tile=fallback.tile(),
      output_tiles_per_workgroup=1,
    )
    let runtime = projection_build_runtime_dispatch(parameters, 1024, target, [
      row_variant_test_record(parameters, scalar),
    ])
    assert_eq(runtime.select(8).source(), OfflineAutotune)
    let (source, variants) = projection_row_variant_sources(
      parameters,
      row_variant_test_profile(),
      runtime,
      runtime.dispatch_plan(1024),
      "scalar",
      max_registers=128,
    )
    assert_eq(source, "")
    assert_true(variants.is_empty())
  }
}
