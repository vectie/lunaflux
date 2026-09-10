///|
test "sibling row product is partially evaluated from the immutable token bound" {
  let dispatch = projection_compiler_dispatch(GatedMlp)
  for
    row_case in [
      (2, 16),
      (8, 16),
      (16, 16),
      (17, 32),
      (32, 32),
      (33, 48),
      (48, 48),
      (49, 64),
      (64, 64),
      (128, 64),
      (256, 64),
    ] {
    let (maximum_tokens, expected_rows) = row_case
    let request = ProjectionTileCompilerRequest::new(
      dispatch~,
      maximum_rows=Int::min(32, maximum_tokens),
      maximum_tokens~,
      demand=AllTokenRows,
    )
    let compilation = compile_projection_tile(request)
    let schedule = compilation.schedule()
    let sibling = schedule.sibling_pipeline().unwrap()
    assert_eq(sibling.row_width(), expected_rows)
    assert_eq(sibling.tile_width(), 16)
    assert_eq(sibling.tile_count(), 64)
    assert_eq(sibling.transfer_width(), 32)
    assert_eq(sibling.stage_count(), 2)
    assert_eq(sibling.minimum_rows(), 2)
    assert_eq(sibling.output_tiles(), 4)
    assert_eq(compilation, compile_projection_tile(request))
    // Specialization bounds independent rows, never reduction terms.
    assert_eq(
      schedule.loops()[4].components(),
      (ProjectionSequentialFold, ProjectionIntermediateReductionAxis, 3072, 16),
    )
  }
}

///|
test "sibling row demand preserves shape-neutral reduction and output geometry" {
  for width in [256, 512, 1024] {
    let dispatch = @luna_projection_strategy.plan_projection_dispatch(
      @luna_projection_strategy.ProjectionShape::new(
        GatedMlp,
        input_width=width,
        output_width=128,
        intermediate_width=512,
      ),
      128,
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
    for maximum_tokens in [8, 16, 32, 64] {
      let compilation = compile_projection_tile(
        ProjectionTileCompilerRequest::new(
          dispatch~,
          maximum_rows=Int::min(32, maximum_tokens),
          maximum_tokens~,
          demand=AllTokenRows,
        ),
      )
      let pipeline = compilation.schedule().sibling_pipeline().unwrap()
      assert_eq(pipeline.row_width(), Int::max(16, maximum_tokens))
      assert_eq(pipeline.tile_count() * pipeline.tile_width(), width)
      assert_eq(pipeline.transfer_width(), 32)
    }
  }
}
