///|
test "QKV narrow output maps retain bounded row variants and separate weights" {
  let parameters : ProjectionParameters = {
    family: QkvProjection,
    input_width: 1024,
    output_width: 4096,
    query_width: 2048,
    intermediate_width: 1024,
  }
  let target = projection_fixture_target()
  let baseline = projection_build_runtime_dispatch(parameters, 1024, target, [])
  let primary = baseline.dispatch_plan(1024)
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
      fail("missing QKV output map")
    }
    let runtime = projection_build_runtime_dispatch(parameters, 1024, target, [
      row_variant_test_record(parameters, selected),
    ])
    assert_eq(runtime.select(1).strategy(), baseline.select(1).strategy())
    assert_eq(runtime.select(9).strategy(), baseline.select(9).strategy())
    assert_eq(runtime.select(1).source(), StaticFallback)
    assert_eq(runtime.select(9).source(), StaticFallback)
    assert_eq(runtime.dispatch_plan(1024), primary)
    let (source, variants) = projection_row_variant_sources(
      parameters,
      row_variant_test_profile(),
      runtime,
      primary,
      "qkv",
      max_registers=128,
    )
    assert_eq(variants.length(), 2)
    let variant = variants[0]
    assert_eq(variant.maximum_token_rows(), 8)
    assert_eq(variant.function_symbol(), "qkv_rows8")
    assert_eq(variant.dimensions().block_x(), groups * 32)
    assert_eq(variant.dimensions().shared_memory_bytes(), groups * 1024)
    // Static dimensions cover the one-row dot path as well; the prepared
    // matrix bucket reduces grid X to the exact output-tile count below.
    assert_eq(variant.dimensions().grid_x(), 4096 / groups)
    assert_true(source.contains("const int column_groups ="))
    assert_true(source.contains("token_count > 8"))
    assert_true(source.contains("const __nv_bfloat16 *query_weight"))
    assert_true(source.contains("const __nv_bfloat16 *key_weight"))
    assert_true(source.contains("const __nv_bfloat16 *value_weight"))
    assert_true(source.contains("selected_weight = value_weight"))
    assert_true(source.contains("selected_weight = key_weight"))
    assert_true(source.contains("k += 16"))
    assert_true(source.contains("mma.sync.aligned.m16n8k16"))
    assert_false(source.contains("atomicAdd"))
    assert_false(source.contains("cudaMalloc"))
    if @env.get_env_var("LUNA_TEST_EXPORT_QKV_DISTRIBUTION") == Some("1") {
      println("LUNA_SOURCE_BEGIN qkv-\{groups}-8-in-1024")
      println(row_variant_primary_test_source(parameters, primary))
      println(source)
      println("LUNA_SOURCE_END")
    }
  }
}
