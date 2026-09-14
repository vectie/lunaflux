///|
test "multi stage lowering keeps publication uniform and drains asynchronous tails" {
  let parameters = sibling_resource_test_parameters()
  let runtime = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  )
  for stages in [3, 4] {
    let request = @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
      dispatch=runtime.dispatch_plan(1024),
      maximum_rows=32,
      maximum_tokens=1024,
      demand=AllTokenRows,
    ).with_fold_pipeline(
      IntermediateFold,
      stages~,
      transfer_tiles=4,
      fragment_stages=2,
    )
    let pipeline = @luna_projection_tile_compiler.compile_projection_tile(
        request,
      )
      .schedule()
      .intermediate_pipeline()
      .unwrap()
    for asynchronous in [true, false] {
      let source = projection_fold_lifetime_source(
        pipeline,
        "consume_current_slot();",
        asynchronous~,
      )
      assert_true(source.contains("prime < \{stages - 1}"))
      assert_true(source.contains("stage = (stage + 1) % \{stages}"))
      assert_eq(source.split("__syncthreads();").count(), 3)
      let pieces = source.split("consume_current_slot();").to_array()
      assert_eq(pieces.length(), 2)
      let future = "stage_operands(inner + \{(stages - 1) * 64}"
      if asynchronous {
        assert_true(pieces[0].contains(future))
        assert_false(pieces[1].contains(future))
        assert_true(source.contains("cp.async.wait_group \{stages - 2};"))
        assert_true(pieces[1].contains("cp.async.wait_group 0;"))
      } else {
        assert_false(pieces[0].contains(future))
        assert_true(pieces[1].contains(future))
        assert_false(source.contains("cp.async"))
      }
    }
  }
}
