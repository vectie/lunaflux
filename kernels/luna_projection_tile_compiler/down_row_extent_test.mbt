///|
test "materialized down fold specializes row demand without changing reduction" {
  let dispatch = @luna_projection_strategy.plan_projection_dispatch(
    @luna_projection_strategy.ProjectionShape::new(
      GatedMlp,
      input_width=256,
      output_width=256,
      intermediate_width=512,
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
  for
    row_case in [
      (1, 16),
      (2, 16),
      (4, 16),
      (8, 16),
      (16, 16),
      (17, 32),
      (32, 32),
      (33, 32),
      (48, 32),
      (49, 32),
      (64, 32),
      (65, 32),
      (1024, 32),
    ] {
    let (maximum_tokens, expected_rows) = row_case
    let compiled = compile_projection_tile(
      ProjectionTileCompilerRequest::new(
        dispatch~,
        maximum_rows=Int::min(maximum_tokens, 8),
        maximum_tokens~,
        demand=AllTokenRows,
      ),
    )
    let pipeline = compiled.schedule().intermediate_pipeline().unwrap()
    assert_eq(pipeline.row_width(), expected_rows)
    assert_eq(pipeline.tile_width(), 16)
    assert_eq(pipeline.tile_count(), 32)
    assert_eq(pipeline.transfer_width(), 64)
    assert_eq(pipeline.stage_count(), 2)
    assert_eq(pipeline.minimum_rows(), 2)
    assert_eq(compiled.schedule().single_row_reduction(), OrderedDot)
    assert_eq(
      compiled.schedule().loops()[4].components(),
      (ProjectionSequentialFold, ProjectionIntermediateReductionAxis, 512, 16),
    )
    // Demand pruning acts only on the independent row map. Every live row
    // still owns exactly one ordered traversal of all 32 reduction tiles.
    for live in [1, maximum_tokens] {
      let visits = FixedArray::make(live, 0)
      let group_count = 1 + (live - 1) / expected_rows
      for group in 0..<group_count {
        let start = group * expected_rows
        for tile_row in 0..<(expected_rows / 16) {
          if start + tile_row * 16 >= live {
            break
          }
          for element in 0..<16 {
            let row = start + tile_row * 16 + element
            if row < live {
              for reduction_tile in 0..<pipeline.tile_count() {
                assert_eq(visits[row], reduction_tile)
                visits[row] += 1
              }
            }
          }
        }
      }
      for visited in visits {
        assert_eq(visited, pipeline.tile_count())
      }
    }
  }
}
