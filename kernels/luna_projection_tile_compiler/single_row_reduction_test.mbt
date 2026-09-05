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
  let request = ProjectionTileCompilerRequest::new(
    dispatch~,
    maximum_rows=1,
    maximum_tokens=16,
    demand=AllTokenRows,
    single_row_numerics=AllowDeterministicDotTree,
  )
  let staged = schedule_projection_tile(
    optimize_projection_tile(
      elaborate_projection_tile(select_projection_tile(request)),
    ),
  )
  assert_eq(staged, compile_projection_tile(request))
  assert_eq(staged.schedule().single_row_reduction(), StridedPairwiseDot(64))
}
