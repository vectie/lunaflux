///|
test "bounded row maps remove only wholly dead matrix microtiles" {
  for bound in [1, 2, 4, 8, 15, 16, 17, 31, 32, 65, 1024] {
    for preferred in [16, 32, 64] {
      let extent = projection_bounded_row_extent(
        maximum_rows=bound,
        matrix_rows=16,
        preferred_rows=preferred,
      ).unwrap()
      assert_eq(extent % 16, 0)
      assert_true(extent <= preferred)
      assert_true(extent >= Int::min(bound, preferred))
      assert_true(extent - 16 < Int::min(bound, preferred))
    }
  }
  assert_eq(
    projection_bounded_row_extent(
      maximum_rows=2147483647,
      matrix_rows=16,
      preferred_rows=64,
    ),
    Some(64),
  )
  for
    invalid in [(0, 16, 32), (-1, 16, 32), (8, 0, 32), (8, 16, 0), (8, 16, 31)] {
    let (maximum_rows, matrix_rows, preferred_rows) = invalid
    assert_eq(
      projection_bounded_row_extent(
        maximum_rows~,
        matrix_rows~,
        preferred_rows~,
      ),
      None,
    )
  }
}

///|
test "dense projection row storage follows token bounds not request slots" {
  for family in [@luna_projection_strategy.ProjectionSemanticFamily::Qkv, Dense] {
    for bound in [1, 2, 4, 8, 16, 17, 32, 65, 1024] {
      let dispatch = @luna_projection_strategy.plan_projection_dispatch(
        @luna_projection_strategy.ProjectionShape::new(
          family,
          input_width=128,
          output_width=256,
          intermediate_width=0,
        ),
        1024,
        @luna_projection_strategy.ProjectionCapabilities::new(
          subgroup_width=32,
          matrix_tile=Some(
            @luna_projection_strategy.ProjectionTile::new(
              rows=16,
              columns=16,
              reduction=16,
            ),
          ),
        ),
      )
      let compiled = compile_projection_tile(
        ProjectionTileCompilerRequest::new(
          dispatch~,
          maximum_rows=1,
          maximum_tokens=bound,
          demand=AllTokenRows,
        ),
      )
      let pipeline = compiled.schedule().matrix_pipeline().unwrap()
      assert_eq(pipeline.row_width(), if bound <= 16 { 16 } else { 32 })
      assert_eq(pipeline.tile_width(), 16)
      assert_eq(pipeline.tile_count(), 8)
      assert_eq(pipeline.minimum_rows(), 2)
    }
  }
}
