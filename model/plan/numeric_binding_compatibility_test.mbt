///|
test "operation tensor compute type rejects incompatible storage encoding" {
  let operation = numeric_test_embedding_operation()
  let schema = @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors=[
      @numeric_contract.NumericTensorSchema::new(
        ordinal=@numeric_contract.NumericTensorOrdinal::new(0),
        role=@numeric_contract.NumericTensorRole::parameter(),
        shape=@numeric_contract.NumericShape::new([8L, 4L]),
        storage=numeric_test_plain(@numeric_contract.StorageDType::f32()),
      ),
    ],
    operations=[
      @numeric_contract.NumericOperationSchema::new(
        ordinal=@numeric_contract.NumericOperationOrdinal::new(0),
        execution=numeric_test_execution(
          @numeric_contract.ComputeDType::absent(),
          @numeric_contract.ComputeDType::bfloat16(),
        ),
      ),
    ],
  )
  try
    @plan.ModelPlan::new_stateless(
      content_digest=numeric_test_content(),
      limits=@plan.ModelPlanLimits::production(),
      shape_constraints=numeric_test_constraints(),
      kv_cache_geometry=numeric_test_geometry(),
      workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
      operations=[operation],
      final_output=@plan.ActivationRef::new(0),
      numeric_schema=schema,
    )
  catch {
    @plan.InvalidGraph(NumericTensorType) => ()
    _ => fail("unexpected tensor-storage failure")
  } noraise {
    _ => fail("incompatible tensor storage was admitted")
  }
}

///|
test "operation tensor shape must match its exact semantic position" {
  let operation = numeric_test_embedding_operation()
  let schema = @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors=[
      @numeric_contract.NumericTensorSchema::new(
        ordinal=@numeric_contract.NumericTensorOrdinal::new(0),
        role=@numeric_contract.NumericTensorRole::parameter(),
        shape=@numeric_contract.NumericShape::new([4L, 8L]),
        storage=numeric_test_plain(@numeric_contract.StorageDType::bfloat16()),
      ),
    ],
    operations=[
      @numeric_contract.NumericOperationSchema::new(
        ordinal=@numeric_contract.NumericOperationOrdinal::new(0),
        execution=numeric_test_execution(
          @numeric_contract.ComputeDType::absent(),
          @numeric_contract.ComputeDType::bfloat16(),
        ),
      ),
    ],
  )
  try
    @plan.ModelPlan::new_stateless(
      content_digest=numeric_test_content(),
      limits=@plan.ModelPlanLimits::production(),
      shape_constraints=numeric_test_constraints(),
      kv_cache_geometry=numeric_test_geometry(),
      workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
      operations=[operation],
      final_output=@plan.ActivationRef::new(0),
      numeric_schema=schema,
    )
  catch {
    @plan.InvalidGraph(NumericTensorShape) => ()
    _ => fail("unexpected tensor-shape failure")
  } noraise {
    _ => fail("transposed embedding storage shape was admitted")
  }
}

///|
test "numeric tensor table rejects unused parameters and orphan metadata" {
  let operation = numeric_test_embedding_operation()
  let execution = numeric_test_execution(
    @numeric_contract.ComputeDType::absent(),
    @numeric_contract.ComputeDType::bfloat16(),
  )
  let parameter = @numeric_contract.NumericTensorSchema::new(
    ordinal=@numeric_contract.NumericTensorOrdinal::new(0),
    role=@numeric_contract.NumericTensorRole::parameter(),
    shape=@numeric_contract.NumericShape::new([8L, 4L]),
    storage=numeric_test_plain(@numeric_contract.StorageDType::bfloat16()),
  )
  let operation_schema = @numeric_contract.NumericOperationSchema::new(
    ordinal=@numeric_contract.NumericOperationOrdinal::new(0),
    execution~,
  )
  let unused_parameter = @numeric_contract.NumericTensorSchema::new(
    ordinal=@numeric_contract.NumericTensorOrdinal::new(1),
    role=@numeric_contract.NumericTensorRole::parameter(),
    shape=@numeric_contract.NumericShape::new([4L]),
    storage=numeric_test_plain(@numeric_contract.StorageDType::bfloat16()),
  )
  let unused_schema = @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors=[parameter, unused_parameter],
    operations=[operation_schema],
  )
  try
    @plan.ModelPlan::new_stateless(
      content_digest=numeric_test_content(),
      limits=@plan.ModelPlanLimits::production(),
      shape_constraints=numeric_test_constraints(),
      kv_cache_geometry=numeric_test_geometry(),
      workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
      operations=[operation],
      final_output=@plan.ActivationRef::new(0),
      numeric_schema=unused_schema,
    )
  catch {
    @plan.InvalidGraph(UnusedParameterTensor) => ()
    _ => fail("unexpected unused-parameter failure")
  } noraise {
    _ => fail("unused parameter tensor was admitted")
  }
  let metadata = @numeric_contract.NumericTensorSchema::new(
    ordinal=@numeric_contract.NumericTensorOrdinal::new(1),
    role=@numeric_contract.NumericTensorRole::scale_metadata(),
    shape=@numeric_contract.NumericShape::new([]),
    storage=numeric_test_plain(@numeric_contract.StorageDType::f32()),
  )
  let metadata_schema = @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors=[parameter, metadata],
    operations=[operation_schema],
  )
  try
    @plan.ModelPlan::new_stateless(
      content_digest=numeric_test_content(),
      limits=@plan.ModelPlanLimits::production(),
      shape_constraints=numeric_test_constraints(),
      kv_cache_geometry=numeric_test_geometry(),
      workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
      operations=[operation],
      final_output=@plan.ActivationRef::new(0),
      numeric_schema=metadata_schema,
    )
  catch {
    @plan.InvalidGraph(UnreferencedNumericMetadata) => ()
    _ => fail("unexpected orphan-metadata failure")
  } noraise {
    _ => fail("unreferenced numeric metadata was admitted")
  }
}

///|
test "one parameter may be reused at compatible semantic positions" {
  let embedding = numeric_test_embedding_operation()
  let head = @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(1),
    kind=LanguageModelHead,
    inputs=[
      Activation(@plan.ActivationRef::new(0)),
      Tensor(@plan.TensorRef::new(0)),
    ],
    outputs=[@plan.ActivationRef::new(1)],
    shape=ProjectionShape(input_width=4, output_width=8),
    required_capability=@plan.KernelCapabilityId::dense_projection(),
    workspace_bytes=0L,
  )
  let schema = @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors=[
      @numeric_contract.NumericTensorSchema::new(
        ordinal=@numeric_contract.NumericTensorOrdinal::new(0),
        role=@numeric_contract.NumericTensorRole::parameter(),
        shape=@numeric_contract.NumericShape::new([8L, 4L]),
        storage=numeric_test_plain(@numeric_contract.StorageDType::bfloat16()),
      ),
    ],
    operations=[
      @numeric_contract.NumericOperationSchema::new(
        ordinal=@numeric_contract.NumericOperationOrdinal::new(0),
        execution=numeric_test_execution(
          @numeric_contract.ComputeDType::absent(),
          @numeric_contract.ComputeDType::bfloat16(),
        ),
      ),
      @numeric_contract.NumericOperationSchema::new(
        ordinal=@numeric_contract.NumericOperationOrdinal::new(1),
        execution=numeric_test_execution(
          @numeric_contract.ComputeDType::bfloat16(),
          @numeric_contract.ComputeDType::bfloat16(),
        ),
      ),
    ],
  )
  let plan = @plan.ModelPlan::new_stateless(
    content_digest=numeric_test_content(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=numeric_test_constraints(),
    kv_cache_geometry=numeric_test_geometry(),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations=[embedding, head],
    final_output=@plan.ActivationRef::new(1),
    numeric_schema=schema,
  )
  assert_eq(plan.tensor_count(), 1)
}
