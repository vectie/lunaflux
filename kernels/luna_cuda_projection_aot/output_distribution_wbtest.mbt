///|
test "dense output C8 distribution preserves exact shape and larger-row fallback" {
  // The attention output need not have the hidden input width. Exercise the
  // rectangular K2048/N1024 shape, not the older square projection fixture.
  let parameters : ProjectionParameters = {
    family: DenseProjection,
    input_width: 2048,
    output_width: 1024,
    query_width: 0,
    intermediate_width: 0,
  }
  let target = projection_fixture_target()
  let baseline = projection_build_runtime_dispatch(parameters, 1024, target, [])
  let primary = baseline.dispatch_plan(1024)
  let profile = row_variant_test_profile()
  for groups in [1, 2, 4] {
    let selected = for
      candidate in baseline
      .dispatch_plan(8)
      .prefill()
      .output_distribution_candidates() {
      if candidate.output_tiles_per_workgroup() == groups {
        break candidate
      }
    } nobreak {
      fail("missing dense output distribution")
    }
    let runtime = projection_build_runtime_dispatch(parameters, 1024, target, [
      row_variant_test_record(parameters, selected),
    ])
    assert_eq(runtime.select(1).source(), StaticFallback)
    assert_eq(runtime.select(8).source(), OfflineAutotune)
    assert_eq(runtime.select(9).source(), StaticFallback)
    assert_eq(runtime.select(1024).source(), StaticFallback)
    let (source, variants) = projection_row_variant_sources(
      parameters,
      profile,
      runtime,
      primary,
      "output",
      max_registers=128,
    )
    assert_eq(variants.length(), 2)
    let variant = variants[0]
    assert_eq(variant.maximum_token_rows(), 8)
    assert_eq(variant.function_symbol(), "output_rows8")
    assert_eq(variant.dimensions().block_x(), groups * 32)
    assert_eq(variant.dimensions().shared_memory_bytes(), groups * 1024)
    // The manifest envelope includes the single-row branch. The prepared
    // matrix bucket derives 64/groups CTAs from its bounded launch contract.
    assert_eq(variant.dimensions().grid_x(), 1024 / groups)
    assert_true(source.contains("group_count = \{64 / groups}"))
    assert_true(source.contains("token_count > 8"))
    assert_true(source.contains("inner < 2048"))
    assert_true(source.contains("k += 16"))
    let compilation = @luna_projection_tile_compiler.compile_projection_tile(
      @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
        dispatch=runtime.dispatch_plan(8),
        maximum_rows=8,
        maximum_tokens=8,
        demand=AllTokenRows,
      ),
    )
    let pipeline = compilation.schedule().matrix_pipeline().unwrap()
    assert_eq(pipeline.transfer_width(), if groups <= 2 { 256 } else { 64 })
    assert_eq(pipeline.tile_width(), 16)
    assert_eq(pipeline.tile_count(), 128)
    assert_eq(pipeline.row_width(), 16)
    assert_eq(projection_workspace_bytes(parameters, profile, primary), 0L)
    if @env.get_env_var("LUNA_TEST_EXPORT_OUTPUT_DISTRIBUTION") == Some("1") {
      println("LUNA_SOURCE_BEGIN output-g\{groups}")
      println(row_variant_primary_test_source(parameters, primary))
      println(source)
      println("LUNA_SOURCE_END")
    }
  }
  if @env.get_env_var("LUNA_TEST_EXPORT_OUTPUT_DISTRIBUTION") == Some("1") {
    println("LUNA_SOURCE_BEGIN output-g8")
    println(row_variant_primary_test_source(parameters, primary))
    println("LUNA_SOURCE_END")
  }
}
