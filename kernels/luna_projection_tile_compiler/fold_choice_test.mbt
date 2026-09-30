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
  let global = ProjectionFoldChoice::new(
    SiblingFold,
    stages=2,
    transfer_tiles=2,
    fragment_stages=1,
    row_tiles=4,
    output_tiles=4,
  )
  let small = ProjectionFoldChoice::new(
    SiblingFold,
    stages=2,
    transfer_tiles=2,
    fragment_stages=1,
    row_tiles=1,
    output_tiles=1,
    consumer_groups=1,
    maximum_token_rows=16,
  )
  assert_eq(
    compile_projection_tile(request.with_fold_choices([global, small])),
    compile_projection_tile(request.with_fold_choices([global])),
  )
  let bounded = ProjectionTileCompilerRequest::new(
    dispatch~,
    maximum_rows=16,
    maximum_tokens=16,
    demand=AllTokenRows,
  )
  assert_eq(
    compile_projection_tile(bounded.with_fold_choices([global, small])),
    compile_projection_tile(bounded.with_fold_choices([small])),
  )
  assert_eq(
    compile_projection_tile(bounded.with_fold_choices([global, small])),
    compile_projection_tile(bounded.with_fold_choices([small, global])),
  )
  let sibling_geometry = request.with_fold_pipeline(
    SiblingFold,
    stages=2,
    transfer_tiles=2,
    fragment_stages=1,
    row_tiles=4,
    output_tiles=4,
  )
  let sibling_plan = compile_projection_tile(sibling_geometry)
  assert_eq(sibling_plan.program().digest(), baseline.program().digest())
  assert_eq(sibling_plan.schedule().sibling_pipeline().unwrap().row_width(), 64)
  assert_eq(
    sibling_plan.schedule().sibling_pipeline().unwrap().output_tiles(),
    4,
  )
  let refined_down = compile_projection_tile(
    request.with_fold_pipeline(
      IntermediateFold,
      stages=2,
      transfer_tiles=2,
      fragment_stages=1,
      row_tiles=4,
      output_tiles=1,
      consumer_groups=1,
    ),
  )
  assert_eq(
    refined_down.schedule().companion_output_tiles_per_workgroup(),
    Some(1),
  )
  assert_eq(refined_down.program().digest(), baseline.program().digest())
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
