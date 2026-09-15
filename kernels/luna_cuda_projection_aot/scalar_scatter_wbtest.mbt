///|
test "scalar selected-row projection scatters to original token rows" {
  let compilation = projection_fixture_lowering(6).tile_compilation()
  assert_eq(compilation.schedule().demand(), QueryRowEnds)
  let parameters : ProjectionParameters = {
    family: LanguageModelHead,
    input_width: 4,
    output_width: 16,
    query_width: 0,
    intermediate_width: 0,
  }
  let source = @utf8.decode(
    projection_scalar_map_source(
      "selected_head_probe", parameters, 4, 2, compilation,
    ),
  )
  assert_true(source.contains("row_offsets[logical_row + 1] - 1"))
  assert_true(
    source.contains("output[(unsigned long long)row * 16ULL + column]"),
  )
  assert_false(source.contains("output[index]"))
  let dense = @utf8.decode(projection_fixture_lowering(4).source_bytes())
  assert_true(dense.contains("output[index]"))
  if @env.get_env_var("LUNA_TEST_EXPORT_SCALAR_SCATTER") == Some("1") {
    println("LUNA_SOURCE_BEGIN scalar-scatter")
    println(source)
    println("LUNA_SOURCE_END")
  }
}
