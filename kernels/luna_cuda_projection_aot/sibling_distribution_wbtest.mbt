///|
test "sibling lowering distributes independent row folds across launched consumers" {
  let parameters : ProjectionParameters = {
    family: GatedMlp,
    input_width: 1024,
    output_width: 1024,
    query_width: 0,
    intermediate_width: 3072,
  }
  let runtime = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  )
  let source = row_variant_primary_test_source(
    parameters,
    runtime.dispatch_plan(1024),
  )
  assert_true(source.contains("const int sibling = warp / 8;"))
  assert_true(source.contains("const int consumer_row = (consumer / 4) * 2;"))
  assert_true(source.contains("const int consumer_column = consumer % 4;"))
  assert_true(source.contains("float> c[2];"))
  assert_false(source.contains("float> cg[1], cu[1];"))
  assert_true(source.contains("if (warp < 16)"))
  assert_true(source.contains("(consumer_row + row) * 16 >= token_count"))
  assert_true(source.contains("column_begin + (tile % 4) * 16 + cell % 16"))
  // Keep copies and CTA publication uniform even for padded tail row owners.
  assert_true(source.contains("cp.async.wait_group 0;"))
  assert_true(
    source.contains(
      "float *gate_tiles = reinterpret_cast<float *>(sibling_storage);",
    ),
  )
  assert_true(
    source.contains(
      "float *up_tiles = reinterpret_cast<float *>(sibling_output_bytes);",
    ),
  )
  if @env.get_env_var("LUNA_TEST_EXPORT_SIBLING_DISTRIBUTION") == Some("1") {
    println("LUNA_SOURCE_BEGIN gate_up")
    println(source)
    println("LUNA_SOURCE_END")
  }
}
