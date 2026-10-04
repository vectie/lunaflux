///|
test "bounded down distribution changes neither producer nor unbounded program" {
  let parameters = sibling_resource_test_parameters()
  let runtime = projection_build_runtime_dispatch(
    parameters,
    2048,
    projection_fixture_target(),
    [],
  )
  let dispatch = runtime.dispatch_plan(2048)
  let inherited = [
    @luna_projection_tile_compiler.ProjectionFoldChoice::new(
      IntermediateFold,
      stages=2,
      transfer_tiles=4,
      fragment_stages=1,
      row_tiles=8,
      hoist_addresses=true,
    ),
    @luna_projection_tile_compiler.ProjectionFoldChoice::new(
      SiblingFold,
      stages=2,
      transfer_tiles=4,
      fragment_stages=1,
      row_tiles=1,
      output_tiles=2,
      consumer_groups=2,
      maximum_token_rows=16,
    ),
  ]
  let choices = [
    ..inherited,
    @luna_projection_tile_compiler.ProjectionFoldChoice::new(
      IntermediateFold,
      stages=2,
      transfer_tiles=8,
      fragment_stages=1,
      row_tiles=1,
      output_tiles=1,
      consumer_groups=1,
      hoist_addresses=true,
      maximum_token_rows=16,
    ),
  ]
  for bound in [8, 16, 32, 2048] {
    let before = projection_refine_physical(
      projection_row_variant_compilation(
        parameters,
        dispatch,
        Int::min(32, bound),
        bound,
        fold_choices=inherited,
      ),
      sibling_execution=OverlappedTransfer,
    )
    let after = projection_refine_physical(
      projection_row_variant_compilation(
        parameters,
        dispatch,
        Int::min(32, bound),
        bound,
        fold_choices=choices,
      ),
      sibling_execution=OverlappedTransfer,
    )
    assert_eq(before.sibling, after.sibling)
    assert_eq(before.sibling_epilogue, after.sibling_epilogue)
    assert_eq(before.down_scalar, after.down_scalar)
    assert_eq(before.ordered_down, after.ordered_down)
    if bound > 16 {
      assert_eq(before, after)
    } else {
      let selected = after.intermediate.unwrap()
      assert_eq(selected.schedule.row_width(), 16)
      assert_eq(selected.schedule.transfer_width(), 128)
      assert_eq(selected.schedule.output_tiles(), 1)
    }
  }
}
