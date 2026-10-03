///|
test "matrix bootstrap and future copies share the retained transfer effect" {
  let parameters : ProjectionParameters = {
    family: DenseProjection,
    input_width: 2048,
    output_width: 1024,
    query_width: 0,
    intermediate_width: 0,
  }
  let dispatch = projection_build_runtime_dispatch(
    parameters,
    2048,
    projection_fixture_target(),
    [],
  ).dispatch_plan(2048)
  for hoist in [false, true] {
    for transfer_tiles in [2, 4] {
      let request = @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
        dispatch~,
        maximum_rows=32,
        maximum_tokens=2048,
        demand=AllTokenRows,
      ).with_fold_pipeline(
        MatrixFold,
        stages=2,
        transfer_tiles~,
        fragment_stages=1,
        row_tiles=4,
        output_tiles=4,
        hoist_addresses=hoist,
      )
      let physical = projection_refine_physical(
        @luna_projection_tile_compiler.compile_projection_tile(request),
      )
      let source = projection_matrix_pipeline_source(parameters, physical, 32)
      let producer = source.split("auto stage_operands").to_array()[1]
        .split("float c[")
        .to_array()[0]
      let pending = physical.matrix.unwrap().effects.has_pending_transfers()
      assert_eq(pending, transfer_tiles == 4)
      assert_eq(producer.contains("cp.async.cg.shared.global"), pending)
      assert_eq(producer.contains("ld.global.v4.u32"), !pending)
      assert_false(source.contains("bool overlap"))
      assert_false(source.contains("if (overlap)"))
      assert_true(source.contains("stage_operands(0, 0);"))
      assert_eq(source.contains("cp.async.wait_group 0;"), pending)
    }
  }
}
