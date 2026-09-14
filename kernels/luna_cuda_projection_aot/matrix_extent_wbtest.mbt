///|
test "matrix column ceiling does not overflow for large legal output widths" {
  for output_width in [256, 3072, 2147483632] {
    let parameters : ProjectionParameters = {
      family: DenseProjection,
      input_width: 128,
      output_width,
      query_width: 0,
      intermediate_width: 0,
    }
    let runtime = projection_build_runtime_dispatch(
      parameters,
      16,
      projection_fixture_target(),
      [],
    )
    let compilation = @luna_projection_tile_compiler.compile_projection_tile(
      @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
        dispatch=runtime.dispatch_plan(16),
        maximum_rows=1,
        maximum_tokens=16,
        demand=AllTokenRows,
        single_row_numerics=AllowDeterministicDotTree,
      ),
    )
    let pipeline = compilation.schedule().matrix_pipeline().unwrap()
    let columns = pipeline.output_tiles().to_int64() * 16L
    let expected = (output_width.to_int64() + columns - 1L) / columns
    let source = projection_matrix_pipeline_source(parameters, compilation, 1)
    assert_true(source.contains("const int column_groups = \{expected};"))
    assert_true(expected > 0L)
  }
}
