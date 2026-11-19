///|
test "public phase pipeline preserves explicit deterministic reduction permission" {
  let dispatch = @luna_projection_strategy.plan_projection_dispatch(
    @luna_projection_strategy.ProjectionShape::new(
      Dense,
      input_width=128,
      output_width=128,
      intermediate_width=0,
    ),
    16,
    @luna_projection_strategy.ProjectionCapabilities::new(
      subgroup_width=64,
      matrix_tile=None,
    ),
  )
  let request = @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
    dispatch~,
    maximum_rows=1,
    maximum_tokens=16,
    demand=AllTokenRows,
    single_row_numerics=AllowDeterministicDotTree,
  )
  let staged = @luna_projection_tile_compiler.schedule_projection_tile(
    @luna_projection_tile_compiler.optimize_projection_tile(
      @luna_projection_tile_compiler.elaborate_projection_tile(
        @luna_projection_tile_compiler.select_projection_tile(request),
      ),
    ),
  )
  assert_eq(
    staged,
    @luna_projection_tile_compiler.compile_projection_tile(request),
  )
  assert_eq(staged.schedule().single_row_reduction(), StridedPairwiseDot(64))
}
