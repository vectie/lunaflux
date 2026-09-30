///|
test "row-scoped column products propagate to source and launch variants only" {
  let parameters : ProjectionParameters = {
    family: DenseProjection,
    input_width: 2048,
    output_width: 1024,
    query_width: 0,
    intermediate_width: 0,
  }
  let runtime = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  )
  let dispatch = runtime.dispatch_plan(1024)
  let choice = @luna_projection_tile_compiler.ProjectionFoldChoice::new(
    MatrixFold,
    stages=2,
    transfer_tiles=4,
    fragment_stages=1,
    row_tiles=1,
    output_tiles=1,
    consumer_groups=1,
    maximum_token_rows=16,
  )
  let primary = projection_row_variant_compilation(
    parameters, dispatch, 32, 1024,
  )
  let scoped = projection_row_variant_compilation(
    parameters,
    dispatch,
    32,
    1024,
    fold_choices=[choice],
  )
  assert_eq(primary, scoped)
  let (source, variants) = projection_row_variant_sources(
    parameters,
    row_variant_test_profile(),
    runtime,
    dispatch,
    "bounded",
    max_registers=128,
    fold_choices=[choice],
  )
  for variant in variants {
    if variant.maximum_token_rows() <= 16 {
      assert_eq(variant.dimensions().block_x(), 32)
      assert_eq(variant.bounded_work_grid_x(), Some(64))
    }
  }
  assert_true(source.contains("bounded_rows16"))
  assert_true(source.contains("const int column_groups = 64;"))
}
