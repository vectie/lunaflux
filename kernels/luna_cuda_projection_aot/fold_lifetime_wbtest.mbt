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
        physical_pipeline_fixture(
          pipeline,
          pipeline.output_tiles(),
          execution=if asynchronous {
            OverlappedTransfer
          } else {
            SerialTransfer
          },
        ),
        projection_test_fold_body(pipeline),
      )
      if stages == 2 {
        assert_true(source.contains("stage_operands(0, 0);"))
        assert_true(source.contains("stage ^= 1;"))
      } else {
        assert_true(source.contains("prime < \{stages - 1}"))
        assert_true(source.contains("stage = (stage + 1) % \{stages}"))
      }
      assert_eq(source.split("__syncthreads();").count(), 3)
      let pieces = source
        .split("const __nv_bfloat16 *current = test_storage")
        .to_array()
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
      let source = fold_pipeline_fixture(
        pipeline, "input", "weight", 3072, 4, 1024,
      )
      assert_eq(source.contains("cp.async"), asynchronous)
      assert_true(source.contains("__syncthreads"))
      let lifetime = projection_fold_lifetime_source(
        physical_pipeline_fixture(pipeline, pipeline.output_tiles()),
        projection_test_fold_body(pipeline),
      )
      let parts = lifetime
        .split("const __nv_bfloat16 *current = test_storage")
        .to_array()
      assert_eq(parts.length(), 2)
      assert_true(parts[0].contains("stage_operands(inner +"))
      assert_false(parts[1].contains("stage_operands(inner +"))
    }
  }
}

///|
fn projection_test_fold_body(
  pipeline : @luna_projection_tile_compiler.ProjectionFoldPipeline,
) -> CudaProjectionFoldBody {
  projection_packed_fold_body(
    physical_pipeline_fixture(pipeline, pipeline.output_tiles()),
    storage="test_storage",
  )
}

///|
test "wide completed physical transfer never emits asynchronous copies" {
  let parameters = sibling_resource_test_parameters()
  let runtime = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  )
  let scheduled = @luna_projection_tile_compiler.compile_projection_tile(
    @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
      dispatch=runtime.dispatch_plan(1024),
      maximum_rows=32,
      maximum_tokens=1024,
      demand=AllTokenRows,
    ).with_fold_pipeline(
      IntermediateFold,
      stages=3,
      transfer_tiles=4,
      fragment_stages=2,
    ),
  )
  let pipeline = scheduled.schedule().intermediate_pipeline().unwrap()
  assert_eq(pipeline.transfer_width(), 64)
  let executions : Array[
    @luna_projection_tile_compiler.ProjectionTransferExecution,
  ] = [SerialTransfer, CompletedPrefetch, OverlappedTransfer]
  for execution in executions {
    let physical = physical_pipeline_fixture(
      pipeline,
      pipeline.output_tiles(),
      execution~,
    )
    let source = projection_fold_pipeline_source(
      physical, "input", "weight", 3072, 1024,
    )
    assert_eq(source.contains("cp.async"), execution == OverlappedTransfer)
    assert_true(source.contains("__syncthreads"))
    if execution != OverlappedTransfer {
      assert_true(source.contains("ld.global.v4.u32"))
      assert_true(source.contains("st.shared.v4.u32"))
    }
  }
}
