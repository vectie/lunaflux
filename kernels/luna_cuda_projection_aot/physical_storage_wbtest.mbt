///|
test "matrix producer and consumer bind one checked storage plan" {
  let parameters = sibling_resource_test_parameters(family=DenseProjection)
  let runtime = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  )
  let compilation = @luna_projection_tile_compiler.compile_projection_tile(
    @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
      dispatch=runtime.dispatch_plan(1024),
      maximum_rows=32,
      maximum_tokens=1024,
      demand=AllTokenRows,
    ),
  )
  let pipeline = compilation.schedule().matrix_pipeline().unwrap()
  let width = pipeline.transfer_width()
  let transport : CudaProjectionTransportLayout = { columns: width, }
  let storage = @physical_tile_ir.OperandStorage::new(
    transport.plan(),
    [pipeline.row_width(), pipeline.output_tiles() * 16],
    owners=128,
    stages=pipeline.stage_count(),
  ).unwrap()
  let body = projection_storage_fold_body(
    pipeline,
    storage,
    rows=1,
    storage="fixture",
    consumers=4,
  )
  assert_true(
    body.source().contains("fixture + stage * \{storage.slot_elements()}"),
  )
  let (weight_offset, _, _) = storage.region(1).unwrap().geometry()
  assert_true(
    body.source().contains("current+\{weight_offset}+0+operand_index"),
  )
}

///|
test "ingress constructor binds complete row-major operand storage" {
  let parameters : ProjectionParameters = {
    family: QkvProjection,
    input_width: 1024,
    output_width: 1024,
    query_width: 512,
    intermediate_width: 256,
  }
  let runtime = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  )
  let compilation = @luna_projection_tile_compiler.compile_projection_tile(
    @luna_projection_tile_compiler.ProjectionTileCompilerRequest::with_attention_ingress_epilogue(
      dispatch=runtime.dispatch_plan(1024),
      maximum_rows=32,
      maximum_tokens=1024,
      epilogue=@luna_projection_tile_compiler.ProjectionAttentionIngressEpilogue::new(
        query_heads=4,
        kv_heads=2,
        head_dimension=128,
        norm_epsilon=0.000001,
        rope_theta=1000000.0,
      ),
    ),
  )
  let pipeline = compilation.schedule().matrix_pipeline().unwrap()
  let width = pipeline.transfer_width()
  let transport : CudaProjectionTransportLayout = { columns: width, }
  let storage = @physical_tile_ir.OperandStorage::new(
    transport.plan(),
    [pipeline.row_width(), pipeline.output_tiles() * 16],
    owners=128,
    stages=pipeline.stage_count(),
  ).unwrap()
  let identity = @physical_tile_ir.OperandLayout::new(
    columns=width,
    vector=8,
    rotations=1,
    row_group=1,
  ).unwrap()
  let ingress_storage = @physical_tile_ir.OperandStorage::new(
    identity,
    [pipeline.row_width(), pipeline.output_tiles() * 16],
    owners=128,
    stages=pipeline.stage_count(),
  ).unwrap()
  let ingress = CudaProjectionFoldBody::ingress(
    pipeline,
    ingress_storage,
    columns=1,
  )
  assert_true(ingress is Some(_))
  assert_true(
    CudaProjectionFoldBody::ingress(pipeline, storage, columns=1) is None,
  )
  for
    (input_rows, weight_rows, stages, columns) in [
      (
        pipeline.row_width() + 1,
        pipeline.output_tiles() * 16,
        pipeline.stage_count(),
        1,
      ),
      (
        pipeline.row_width(),
        pipeline.output_tiles() * 16 + 1,
        pipeline.stage_count(),
        1,
      ),
      (
        pipeline.row_width(),
        pipeline.output_tiles() * 16,
        pipeline.stage_count() + 1,
        1,
      ),
      (
        pipeline.row_width(),
        pipeline.output_tiles() * 16,
        pipeline.stage_count(),
        pipeline.output_tiles() + 1,
      ),
      (
        pipeline.row_width(),
        pipeline.output_tiles() * 16,
        pipeline.stage_count(),
        0,
      ),
    ] {
    let invalid = @physical_tile_ir.OperandStorage::new(
      identity,
      [input_rows, weight_rows],
      owners=128,
      stages~,
    ).unwrap()
    assert_true(
      CudaProjectionFoldBody::ingress(pipeline, invalid, columns~) is None,
    )
  }
}
