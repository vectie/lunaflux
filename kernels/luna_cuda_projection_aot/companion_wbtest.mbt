///|
test "only split matrix MLP emits an independently distributed companion" {
  for width in [8, 1024] {
    let parameters : ProjectionParameters = {
      family: GatedMlp,
      input_width: width,
      output_width: width,
      query_width: 0,
      intermediate_width: width * 3,
    }
    let dispatch = projection_build_runtime_dispatch(
      parameters,
      1024,
      projection_fixture_target(),
      [],
    ).dispatch_plan(1024)
    let compilation = @luna_projection_tile_compiler.compile_projection_tile(
      @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
        dispatch~,
        maximum_rows=32,
        maximum_tokens=1024,
        demand=AllTokenRows,
        single_row_numerics=AllowDeterministicDotTree,
      ),
    )
    let dimensions = projection_companion_dimensions(
      parameters,
      dispatch,
      row_variant_test_profile(),
      compilation,
    )
    if width == 1024 {
      let d = dimensions.unwrap()
      assert_eq(d.block_x(), 128)
      assert_eq(d.shared_memory_bytes(), 4096)
      assert_eq(d.grid_x(), 1024)
      let source = row_variant_primary_test_source(parameters, dispatch)
      assert_true(source.contains("blockDim.x != 128"))
      assert_true(source.contains("void baseline_down("))
    } else {
      assert_eq(dimensions, None)
    }
  }
}
