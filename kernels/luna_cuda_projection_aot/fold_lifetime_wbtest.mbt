///|
test "multi stage lowering keeps publication uniform and drains asynchronous tails" {
  let parameters = sibling_resource_test_parameters()
  let runtime = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  )
  for stages in [2, 3, 4] {
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
      let effects = pipeline.effect_plan(
        if asynchronous {
          OverlappedTransfer
        } else {
          SerialTransfer
        },
      )
      assert_eq(
        projection_fold_bootstrap_source(effects, "issue();", "wait();"),
        if asynchronous {
          "issue();\n      wait();\n      __syncthreads();"
        } else if stages == 2 {
          "issue();\n      __syncthreads();"
        } else {
          "issue();\n      \n      __syncthreads();"
        },
      )
      let source = projection_fold_lifetime_source(
        pipeline,
        "consume_current_slot();",
        asynchronous~,
      )
      if stages == 2 {
        assert_true(source.contains("stage_operands(0, 0);"))
        assert_true(source.contains("stage ^= 1;"))
      } else {
        assert_true(source.contains("prime < \{stages - 1}"))
        assert_true(source.contains("stage = (stage + 1) % \{stages}"))
      }
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

///|
test "completed prefetch eliminates empty async groups but preserves early issue" {
  let parameters = sibling_resource_test_parameters()
  let dispatch = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  ).dispatch_plan(1024)
  for stages in [2, 3, 4] {
    for tiles in [2, 4] {
      let request = @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
        dispatch~,
        maximum_rows=32,
        maximum_tokens=1024,
        demand=AllTokenRows,
      ).with_fold_pipeline(
        IntermediateFold,
        stages~,
        transfer_tiles=tiles,
        fragment_stages=1,
      )
      let pipeline = @luna_projection_tile_compiler.compile_projection_tile(
          request,
        )
        .schedule()
        .intermediate_pipeline()
        .unwrap()
      let asynchronous = projection_operand_copy_is_async(
        pipeline.transfer_width(),
      )
      assert_eq(asynchronous, tiles == 4)
      let source = projection_fold_pipeline_source(
        pipeline, "input", "weight", 3072, 4, 1024,
      )
      assert_eq(source.contains("cp.async"), asynchronous)
      assert_true(source.contains("__syncthreads"))
      let lifetime = projection_fold_lifetime_source(
        pipeline,
        "consume_current();",
        asynchronous=true,
        completed_on_issue=!asynchronous,
      )
      let parts = lifetime.split("consume_current();").to_array()
      assert_eq(parts.length(), 2)
      assert_true(parts[0].contains("stage_operands(inner +"))
      assert_false(parts[1].contains("stage_operands(inner +"))
    }
  }
}
