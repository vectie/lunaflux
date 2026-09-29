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
  let physical = physical_pipeline_fixture(pipeline, 4)
  let storage = physical.operands
  let body = projection_storage_fold_body(
    physical,
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
  let identity = @physical_tile_ir.OperandLayout::new(
    columns=width,
    vector=8,
    rotations=1,
    row_group=1,
  ).unwrap()
  let physical = @luna_projection_tile_compiler.ProjectionPhysicalPipeline::new(
    pipeline,
    identity,
    consumer_groups=4,
    owners_per_group=32,
    products=1,
    OverlappedTransfer,
    columns=1,
  ).unwrap()
  let ingress = CudaProjectionFoldBody::ingress(physical)
  assert_true(ingress is Some(_))
  // Other backends may accept these geometries; CUDA's fixed packed
  // instruction lowering must reject them instead of changing ownership.
  for geometry in [(4, 32), (8, 16), (8, 64)] {
    let (vector, owners_per_group) = geometry
    let layout = @physical_tile_ir.OperandLayout::new(
      columns=width,
      vector~,
      rotations=1,
      row_group=1,
    ).unwrap()
    let incompatible = @luna_projection_tile_compiler.ProjectionPhysicalPipeline::new(
      pipeline,
      layout,
      consumer_groups=4,
      owners_per_group~,
      products=1,
      OverlappedTransfer,
      columns=1,
    ).unwrap()
    assert_false(projection_cuda_fragment_geometry(incompatible))
    assert_true(CudaProjectionFoldBody::ingress(incompatible) is None)
  }
  assert_true(
    CudaProjectionFoldBody::ingress(
      @luna_projection_tile_compiler.ProjectionPhysicalPipeline::new(
        pipeline,
        ({ columns: width, } : CudaProjectionTransportLayout).plan(),
        consumer_groups=4,
        owners_per_group=32,
        products=1,
        OverlappedTransfer,
        columns=1,
      ).unwrap(),
    )
    is None,
  )
  // Geometry is no longer caller-assembled: regions and stage count are
  // derived together. Illegal fragment domains fail before a body exists.
  for columns in [0, pipeline.output_tiles() + 1] {
    let invalid = @luna_projection_tile_compiler.ProjectionPhysicalPipeline::new(
      pipeline,
      identity,
      consumer_groups=4,
      owners_per_group=32,
      products=1,
      OverlappedTransfer,
      columns~,
    )
    assert_true(invalid is None)
  }
}
