///|
test "offline fold choices preserve semantics and compose canonically" {
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
  let request = ProjectionTileCompilerRequest::new(
    dispatch~,
    maximum_rows=32,
    maximum_tokens=1024,
    demand=AllTokenRows,
  )
  let baseline = compile_projection_tile(request)
  for stages in [2, 3, 4] {
    for transfer_tiles in [1, 2, 4] {
      for fragment_stages in 1..<=Int::min(2, transfer_tiles) {
        let down = ProjectionFoldChoice::new(
          IntermediateFold,
          stages~,
          transfer_tiles~,
          fragment_stages~,
        )
        let sibling = ProjectionFoldChoice::new(
          SiblingFold,
          stages~,
          transfer_tiles~,
          fragment_stages~,
        )
        let forward = compile_projection_tile(
          request.with_fold_choices([down, sibling]),
        )
        let reverse = compile_projection_tile(
          request.with_fold_choices([sibling, down]),
        )
        assert_eq(forward.program().digest(), baseline.program().digest())
        assert_eq(forward.schedule().digest(), reverse.schedule().digest())
        let pipeline = forward.schedule().intermediate_pipeline().unwrap()
        assert_eq(pipeline.stage_count(), stages)
        assert_eq(pipeline.transfer_width(), 16 * transfer_tiles)
        assert_eq(pipeline.fragment_stage_count(), fragment_stages)
        assert_eq(pipeline.transfer_count(), 32 / transfer_tiles)
      }
    }
  }
}
