///|
test "long dense output distributions remain independent of bounded row choices" {
  let parameters : ProjectionParameters = {
    family: DenseProjection,
    input_width: 2048,
    output_width: 1024,
    query_width: 0,
    intermediate_width: 0,
  }
  let target = projection_fixture_target()
  let baseline = projection_build_runtime_dispatch(parameters, 1024, target, [])
  for groups in [2, 4, 8] {
    let selected = for
      candidate in baseline
      .dispatch_plan(1024)
      .prefill()
      .output_distribution_candidates() {
      if candidate.output_tiles_per_workgroup() == groups {
        break candidate
      }
    } nobreak {
      fail("missing long output distribution")
    }
    let runtime = projection_build_runtime_dispatch(parameters, 1024, target, [
      row_variant_test_record(parameters, selected),
    ])
    assert_eq(runtime.select(8).strategy(), baseline.select(8).strategy())
    assert_eq(runtime.select(1024).source(), OfflineAutotune)
    let dispatch = runtime.dispatch_plan(1024)
    assert_eq(dispatch.prefill().output_tiles_per_workgroup(), groups)
    let source = row_variant_primary_test_source(parameters, dispatch)
    assert_true(source.contains("k += 16"))
    assert_false(source.contains("atomicAdd"))
    if @env.get_env_var("LUNA_TEST_EXPORT_LONG_OUTPUT") == Some("1") {
      println("LUNA_SOURCE_BEGIN output-long-g\{groups}")
      println(source)
      println("LUNA_SOURCE_END")
    }
  }
}
