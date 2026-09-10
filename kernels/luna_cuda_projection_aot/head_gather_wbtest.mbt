///|
test "selected-row head register gather reaches segmented greedy executable variants" {
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
  for bound in [8, 16, 256] {
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
    let gather = compiled.selected_row_gather().unwrap()
    assert_eq(gather.dimensions(), (16, 16, 16))
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
      "head_segmented_greedy_v2",
      max_registers=128,
    )
    assert_eq(variants.length(), 1)
    let dimensions = variants[0].dimensions()
    assert_eq(dimensions.block_x(), 32)
    assert_eq(dimensions.grid_x(), 9496)
    assert_eq(dimensions.shared_memory_bytes(), 0)
    assert_true(source.contains("head_segmented_greedy_v2_rows\{bound}"))
    assert_true(source.contains("inner < 1024; inner += 16"))
    assert_true(source.contains("token_low >= 0 && token_low < token_count"))
    assert_true(source.contains("token_high >= 0 && token_high < token_count"))
    assert_true(
      source.contains(
        "logical_high < row_count ? row_offsets[logical_high + 1] - 1 : -1",
      ),
    )
    assert_true(source.contains("accumulator[pair * 2]"))
    assert_true(source.contains("mma.sync.aligned.m16n8k16"))
    assert_true(source.contains("if (counts[2] == 1)"))
    assert_true(source.contains("__shfl_down_sync"))
    assert_false(source.contains("__syncthreads"))
    assert_false(source.contains("__shared__"))
    assert_false(source.contains("operand_index"))
    assert_false(source.contains("result_index"))
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
      @luna_projection_tile_compiler.projection_selected_row_gather(
        resident_runtime.dispatch_plan(bound),
        QueryRowEnds,
      )
      is None,
    )
    assert_true(
      @luna_projection_tile_compiler.projection_selected_row_gather(
        dispatch,
        AllTokenRows,
      )
      is None,
    )
    if @env.get_env_var("LUNA_TEST_EXPORT_HEAD_GATHER") == Some("1") {
      let (resident_source, _) = projection_row_variant_sources(
        parameters,
        row_variant_test_profile(),
        resident_runtime,
        primary,
        "resident",
        max_registers=128,
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
        max_registers=128,
      )
      println("LUNA_SOURCE_BEGIN head-gather-\{bound}-in-1024")
      println(row_variant_primary_test_source(parameters, primary))
      println(source)
      println(resident_source)
      println(current_source)
      println("LUNA_SOURCE_END")
    }
  }
}
