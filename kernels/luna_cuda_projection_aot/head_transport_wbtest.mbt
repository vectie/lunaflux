///|
test "selected head transport alternatives retain declared maps and numerical fold" {
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
    for groups in [2, 4] {
      let selected = for
        candidate in baseline
        .dispatch_plan(bound)
        .prefill()
        .output_distribution_candidates() {
        if candidate.output_tiles_per_workgroup() == groups {
          break candidate
        }
      } nobreak {
        fail("head distribution absent")
      }
      let runtime = projection_build_runtime_dispatch(
        parameters,
        1024,
        projection_fixture_target(),
        [row_variant_test_record(parameters, selected)],
      )
      let compiled = @luna_projection_tile_compiler.compile_projection_tile(
        @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
          dispatch=runtime.dispatch_plan(bound),
          maximum_rows=Int::min(32, bound),
          maximum_tokens=bound,
          demand=QueryRowEnds,
        ),
      )
      assert_true(compiled.selected_row_gather() is None)
      assert_true(compiled.schedule().matrix_pipeline() is Some(_))
      assert_eq(
        compiled.schedule().single_row_reduction(),
        @luna_projection_tile_compiler.OrderedDot,
      )
      let (source, variants) = projection_row_variant_sources(
        parameters,
        row_variant_test_profile(),
        runtime,
        baseline.dispatch_plan(1024),
        "head",
        max_registers=128,
      )
      assert_eq(variants.length(), 1)
      assert_eq(variants[0].dimensions().block_x(), groups * 32)
      assert_true(source.contains("ldmatrix.sync.aligned.m8n8.x4.shared.b16"))
      assert_false(source.contains("temporary[result_index("))
      if @env.get_env_var("LUNA_TEST_EXPORT_HEAD_TRANSPORT") == Some("1") {
        println("LUNA_SOURCE_BEGIN head-transport-g\{groups}-b\{bound}")
        println(
          row_variant_primary_test_source(
            parameters,
            baseline.dispatch_plan(1024),
          ),
        )
        println(source)
        println("LUNA_SOURCE_END")
        println(
          "LUNA_LAUNCH head-transport-g\{groups}-b\{bound} \{variants[0].dimensions().grid_x()} \{variants[0].dimensions().block_x()} \{variants[0].dimensions().shared_memory_bytes()}",
        )
      }
    }
  }
}
