///|
test "selected-row head strip is compiled into a bounded executable variant" {
  let parameters : ProjectionParameters = {
    family: LanguageModelHead,
    input_width: 1024,
    output_width: 151936,
    query_width: 0,
    intermediate_width: 0,
  }
  let baseline = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  )
  for bound in [8, 256] {
    let selected = for
      candidate in baseline
      .dispatch_plan(bound)
      .prefill()
      .output_distribution_candidates() {
      if candidate.output_tiles_per_workgroup() == 1 {
        break candidate
      }
    } nobreak {
      fail("missing single-output-tile head strategy")
    }
    let runtime = projection_build_runtime_dispatch(
      parameters,
      1024,
      projection_fixture_target(),
      [row_variant_test_record(parameters, selected)],
    )
    let dispatch = runtime.dispatch_plan(bound)
    let request = @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
      dispatch~,
      maximum_rows=if bound < 32 { bound } else { 32 },
      maximum_tokens=bound,
      demand=QueryRowEnds,
      single_row_numerics=AllowDeterministicDotTree,
    )
    let compiled = @luna_projection_tile_compiler.compile_projection_tile(
      request,
    )
    let strip = compiled.selected_row_strip().unwrap()
    assert_eq(strip.reduction_elements(), 128)
    assert_eq(strip.local_storage_bytes(), 9216)
    assert_eq(
      compiled.schedule().single_row_reduction(),
      @luna_projection_tile_compiler.StridedPairwiseDot(32),
    )
    let primary = baseline.dispatch_plan(1024)
    assert_eq(runtime.dispatch_plan(1024), primary)
    assert_eq(runtime.select(1).strategy(), baseline.select(1).strategy())
    assert_eq(runtime.select(bound + 1).source(), StaticFallback)
    let (source, variants) = projection_row_variant_sources(
      parameters,
      row_variant_test_profile(),
      runtime,
      primary,
      "head",
    )
    assert_eq(variants.length(), 1)
    let dimensions = variants[0].dimensions()
    assert_eq(dimensions.block_x(), 32)
    assert_eq(dimensions.grid_x(), 9496)
    assert_eq(dimensions.shared_memory_bytes(), strip.local_storage_bytes())
    assert_true(source.contains("strip_begin += 128"))
    assert_true(source.contains("inner < 128; inner += 16"))
    assert_true(source.contains("operand_index(local_row, inner + local_k)"))
    assert_true(source.contains("strip_begin + inner + local_k"))
    assert_true(source.contains("token_row >= 0 && token_row < token_count"))
    assert_true(source.contains("if (counts[2] == 1)"))
    assert_true(source.contains("__shfl_down_sync"))
    assert_true(source.contains("__syncthreads"))
    assert_false(source.contains("wmma::"))
    assert_false(source.contains("cp.async.commit_group"))
    assert_false(source.contains("atomicAdd"))
    assert_false(source.contains("cudaMalloc"))
    // A declared row-resident alternative is not silently rewritten.
    let resident = selected.with_selected_row_residency(stable_id=3101)
    let resident_runtime = projection_build_runtime_dispatch(
      parameters,
      1024,
      projection_fixture_target(),
      [row_variant_test_record(parameters, resident)],
    )
    assert_true(
      @luna_projection_tile_compiler.projection_selected_row_strip(
        resident_runtime.dispatch_plan(bound),
        QueryRowEnds,
      )
      is None,
    )
    assert_true(
      @luna_projection_tile_compiler.projection_selected_row_strip(
        dispatch,
        AllTokenRows,
      )
      is None,
    )
    if @env.get_env_var("LUNA_TEST_EXPORT_HEAD_STRIP") == Some("1") {
      let (resident_source, _) = projection_row_variant_sources(
        parameters,
        row_variant_test_profile(),
        resident_runtime,
        primary,
        "resident",
      )
      let current_runtime = projection_build_runtime_dispatch(
        parameters,
        1024,
        projection_fixture_target(),
        [
          row_variant_test_record(
            parameters,
            baseline
            .dispatch_plan(bound)
            .prefill()
            .with_selected_row_residency(stable_id=3108),
          ),
        ],
      )
      let (current_source, _) = projection_row_variant_sources(
        parameters,
        row_variant_test_profile(),
        current_runtime,
        primary,
        "current",
      )
      println("LUNA_SOURCE_BEGIN head-strip-\{bound}-in-1024")
      println(row_variant_primary_test_source(parameters, primary))
      println(source)
      println(resident_source)
      println(current_source)
      println("LUNA_SOURCE_END")
    }
  }
}
