///|
test "pipeline requires complete reduction blocks and shares no model identity" {
  for width in [256, 496, 512, 528, 544, 576, 1024] {
    let dispatch = @luna_projection_strategy.plan_projection_dispatch(
      @luna_projection_strategy.ProjectionShape::new(
        GatedMlp,
        input_width=128,
        output_width=128,
        intermediate_width=width,
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
    let compiled = compile_projection_tile(
      ProjectionTileCompilerRequest::new(
        dispatch~,
        maximum_rows=8,
        maximum_tokens=128,
        demand=AllTokenRows,
      ),
    )
    let pipeline = compiled.schedule().intermediate_pipeline()
    assert_eq(pipeline is Some(_), width >= 256 && width % 64 == 0)
    if pipeline is Some(plan) {
      assert_eq(plan.tile_count() * plan.tile_width(), width)
      assert_eq(plan.row_width(), 32)
      assert_eq(plan.single_row_outputs(), 4)
    }
  }
}
