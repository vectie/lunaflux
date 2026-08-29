///|
fn numeric_test_storage(
  dtype : @numeric_contract.StorageDType,
  encoding : @numeric_contract.StorageEncoding,
  scale : @numeric_contract.ScaleGranularity,
  zero : @numeric_contract.ZeroPointMode,
  codebook : @numeric_contract.CodebookLayout,
  conversion : @numeric_contract.NumericConversion,
) -> @numeric_contract.TensorStorageContract raise {
  @numeric_contract.TensorStorageContract::new(
    dtype~,
    layout=@numeric_contract.StorageLayout::dense_row_major_v1(),
    encoding~,
    scale_granularity=scale,
    zero_point=zero,
    codebook_layout=codebook,
    conversion~,
  )
}

///|
fn numeric_test_plain(
  dtype : @numeric_contract.StorageDType,
) -> @numeric_contract.TensorStorageContract raise {
  numeric_test_storage(
    dtype,
    @numeric_contract.StorageEncoding::plain_v1(),
    @numeric_contract.ScaleGranularity::absent(),
    @numeric_contract.ZeroPointMode::absent(),
    @numeric_contract.CodebookLayout::absent(),
    @numeric_contract.NumericConversion::exact_v1(),
  )
}

///|
fn numeric_test_execution(
  activation : @numeric_contract.ComputeDType,
  tensor : @numeric_contract.ComputeDType,
) -> @numeric_contract.OperationExecutionContract raise {
  let dynamic_fp8 = activation == @numeric_contract.ComputeDType::bfloat16() &&
    tensor == @numeric_contract.ComputeDType::f8_e4m3_finite()
  @numeric_contract.OperationExecutionContract::new(
    activation_input=activation,
    activation_compute=if dynamic_fp8 {
      @numeric_contract.ComputeDType::f8_e4m3_finite()
    } else {
      activation
    },
    activation_scale=if dynamic_fp8 {
      @numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1()
    } else {
      @numeric_contract.ActivationScalePolicy::absent_exact_v1()
    },
    tensor_input=tensor,
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::bfloat16(),
    conversion=if dynamic_fp8 {
      @numeric_contract.NumericConversion::quantized_v1()
    } else {
      @numeric_contract.NumericConversion::exact_v1()
    },
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
}

///|
fn numeric_test_scaled_tensors(
  scale_reference : @numeric_contract.NumericTensorOrdinal,
  fp8 : @numeric_contract.TensorStorageContract,
  scale_storage : @numeric_contract.TensorStorageContract,
  shape : @numeric_contract.NumericShape,
  scalar : @numeric_contract.NumericShape,
) -> Array[@numeric_contract.NumericTensorSchema] raise {
  [
    @numeric_contract.NumericTensorSchema::new(
      ordinal=@numeric_contract.NumericTensorOrdinal::new(0),
      role=@numeric_contract.NumericTensorRole::parameter(),
      shape~,
      storage=fp8,
      scale=@numeric_contract.ScaleTensorOrdinal::new(scale_reference),
    ),
    @numeric_contract.NumericTensorSchema::new(
      ordinal=@numeric_contract.NumericTensorOrdinal::new(1),
      role=@numeric_contract.NumericTensorRole::scale_metadata(),
      shape=scalar,
      storage=scale_storage,
    ),
    @numeric_contract.NumericTensorSchema::new(
      ordinal=@numeric_contract.NumericTensorOrdinal::new(2),
      role=@numeric_contract.NumericTensorRole::scale_metadata(),
      shape=scalar,
      storage=scale_storage,
    ),
  ]
}

///|
fn numeric_test_content() -> @spec.ContentDigest raise {
  @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
}

///|
fn numeric_test_constraints() -> @plan.ShapeConstraints raise {
  @plan.ShapeConstraints::new(
    max_batch_rows=1,
    max_sequence_tokens=8,
    vocabulary_size=8,
    hidden_size=4,
    intermediate_size=8,
    attention_head_count=2,
    kv_head_count=1,
    head_dimension=2,
  )
}

///|
fn numeric_test_geometry() -> @plan.KvCacheGeometry raise {
  @plan.KvCacheGeometry::new(
    layer_count=1,
    kv_head_count=1,
    head_dimension=2,
    dtype=BFloat16,
    layout=LayerMajorSplitKeyValueV1,
  )
}

///|
fn numeric_test_embedding_operation() -> @plan.PlanOperation raise {
  @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(0),
    kind=Embedding,
    inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
    outputs=[@plan.ActivationRef::new(0)],
    shape=EmbeddingShape(vocabulary_size=8, width=4),
    required_capability=@plan.KernelCapabilityId::embedding_lookup(),
    workspace_bytes=0L,
  )
}

///|
test "numeric binding digest covers exact typed metadata references" {
  let fp8 = numeric_test_storage(
    @numeric_contract.StorageDType::f8_e4m3_finite(),
    @numeric_contract.StorageEncoding::scaled_v1(),
    @numeric_contract.ScaleGranularity::per_tensor(),
    @numeric_contract.ZeroPointMode::absent(),
    @numeric_contract.CodebookLayout::absent(),
    @numeric_contract.NumericConversion::quantized_v1(),
  )
  let scale_storage = numeric_test_plain(@numeric_contract.StorageDType::f32())
  let shape = @numeric_contract.NumericShape::new([4L, 4L])
  let scalar = @numeric_contract.NumericShape::new([])
  let operations = [
    @numeric_contract.NumericOperationSchema::new(
      ordinal=@numeric_contract.NumericOperationOrdinal::new(0),
      execution=numeric_test_execution(
        @numeric_contract.ComputeDType::absent(),
        @numeric_contract.ComputeDType::f8_e4m3_finite(),
      ),
    ),
  ]
  let first = @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors=numeric_test_scaled_tensors(
      @numeric_contract.NumericTensorOrdinal::new(1),
      fp8,
      scale_storage,
      shape,
      scalar,
    ),
    operations~,
  )
  let second = @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors=numeric_test_scaled_tensors(
      @numeric_contract.NumericTensorOrdinal::new(2),
      fp8,
      scale_storage,
      shape,
      scalar,
    ),
    operations~,
  )
  assert_false(first.digest() == second.digest())
  assert_false(first.canonical_bytes() == second.canonical_bytes())
}

///|
test "referenced scale metadata remains typed and reachable" {
  let fp8 = numeric_test_storage(
    @numeric_contract.StorageDType::f8_e4m3_finite(),
    @numeric_contract.StorageEncoding::scaled_v1(),
    @numeric_contract.ScaleGranularity::per_tensor(),
    @numeric_contract.ZeroPointMode::absent(),
    @numeric_contract.CodebookLayout::absent(),
    @numeric_contract.NumericConversion::quantized_v1(),
  )
  let schema = @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors=[
      @numeric_contract.NumericTensorSchema::new(
        ordinal=@numeric_contract.NumericTensorOrdinal::new(0),
        role=@numeric_contract.NumericTensorRole::parameter(),
        shape=@numeric_contract.NumericShape::new([8L, 4L]),
        storage=fp8,
        scale=@numeric_contract.ScaleTensorOrdinal::new(
          @numeric_contract.NumericTensorOrdinal::new(1),
        ),
      ),
      @numeric_contract.NumericTensorSchema::new(
        ordinal=@numeric_contract.NumericTensorOrdinal::new(1),
        role=@numeric_contract.NumericTensorRole::scale_metadata(),
        shape=@numeric_contract.NumericShape::new([]),
        storage=numeric_test_plain(@numeric_contract.StorageDType::f32()),
      ),
    ],
    operations=[
      @numeric_contract.NumericOperationSchema::new(
        ordinal=@numeric_contract.NumericOperationOrdinal::new(0),
        execution=numeric_test_execution(
          @numeric_contract.ComputeDType::absent(),
          @numeric_contract.ComputeDType::f8_e4m3_finite(),
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
    operations=[numeric_test_embedding_operation()],
    final_output=@plan.ActivationRef::new(0),
    numeric_schema=schema,
  )
  guard plan.tensor(@plan.TensorRef::new(0)) is Some(tensor) &&
    tensor.scale() is Some(reference) else {
    fail("typed scale reference was lost")
  }
  assert_eq(reference.tensor(), @plan.TensorRef::new(1))
}

///|
test "metadata targets require exact roles and codebook shape" {
  let codebook_source = numeric_test_storage(
    @numeric_contract.StorageDType::i8(),
    @numeric_contract.StorageEncoding::codebook_v1(),
    @numeric_contract.ScaleGranularity::absent(),
    @numeric_contract.ZeroPointMode::absent(),
    @numeric_contract.CodebookLayout::contiguous_f32_values_v1(),
    @numeric_contract.NumericConversion::exact_v1(),
  )
  let plain_f32 = numeric_test_plain(@numeric_contract.StorageDType::f32())
  let tensors = [
    @numeric_contract.NumericTensorSchema::new(
      ordinal=@numeric_contract.NumericTensorOrdinal::new(0),
      role=@numeric_contract.NumericTensorRole::parameter(),
      shape=@numeric_contract.NumericShape::new([4L]),
      storage=codebook_source,
      codebook=@numeric_contract.CodebookTensorOrdinal::new(
        @numeric_contract.NumericTensorOrdinal::new(1),
      ),
    ),
    @numeric_contract.NumericTensorSchema::new(
      ordinal=@numeric_contract.NumericTensorOrdinal::new(1),
      role=@numeric_contract.NumericTensorRole::codebook_metadata(),
      shape=@numeric_contract.NumericShape::new([255L]),
      storage=plain_f32,
    ),
  ]
  try
    @numeric_contract.ModelNumericSchema::new(
      limits=@numeric_contract.ModelNumericSchemaLimits::production(),
      tensors~,
      operations=[],
    )
  catch {
    InvalidNumericContract(StorageCombination) => ()
    _ => fail("unexpected codebook-shape failure")
  } noraise {
    _ => fail("noncanonical codebook shape was admitted")
  }
}

///|
test "metadata tensors cannot be direct operation operands" {
  let operation = @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(0),
    kind=Embedding,
    inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
    outputs=[@plan.ActivationRef::new(0)],
    shape=EmbeddingShape(vocabulary_size=8, width=4),
    required_capability=@plan.KernelCapabilityId::embedding_lookup(),
    workspace_bytes=0L,
  )
  let schema = @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors=[
      @numeric_contract.NumericTensorSchema::new(
        ordinal=@numeric_contract.NumericTensorOrdinal::new(0),
        role=@numeric_contract.NumericTensorRole::scale_metadata(),
        shape=@numeric_contract.NumericShape::new([]),
        storage=numeric_test_plain(@numeric_contract.StorageDType::f32()),
      ),
    ],
    operations=[
      @numeric_contract.NumericOperationSchema::new(
        ordinal=@numeric_contract.NumericOperationOrdinal::new(0),
        execution=numeric_test_execution(
          @numeric_contract.ComputeDType::absent(),
          @numeric_contract.ComputeDType::f32(),
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
    @plan.InvalidGraph(TensorRoleMismatch) => ()
    _ => fail("unexpected metadata-operand failure")
  } noraise {
    _ => fail("metadata tensor was admitted as an operation operand")
  }
}

///|
test "producer and consumer activation precision must agree" {
  let embedding = @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(0),
    kind=Embedding,
    inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
    outputs=[@plan.ActivationRef::new(0)],
    shape=EmbeddingShape(vocabulary_size=8, width=4),
    required_capability=@plan.KernelCapabilityId::embedding_lookup(),
    workspace_bytes=0L,
  )
  let norm = @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(1),
    kind=RmsNorm,
    inputs=[
      Activation(@plan.ActivationRef::new(0)),
      Tensor(@plan.TensorRef::new(1)),
    ],
    outputs=[@plan.ActivationRef::new(1)],
    shape=NormalizationShape(width=4, epsilon=0.000001),
    required_capability=@plan.KernelCapabilityId::rms_norm(),
    workspace_bytes=0L,
  )
  let storage = numeric_test_plain(@numeric_contract.StorageDType::bfloat16())
  let schema = @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors=[
      @numeric_contract.NumericTensorSchema::new(
        ordinal=@numeric_contract.NumericTensorOrdinal::new(0),
        role=@numeric_contract.NumericTensorRole::parameter(),
        shape=@numeric_contract.NumericShape::new([8L, 4L]),
        storage~,
      ),
      @numeric_contract.NumericTensorSchema::new(
        ordinal=@numeric_contract.NumericTensorOrdinal::new(1),
        role=@numeric_contract.NumericTensorRole::parameter(),
        shape=@numeric_contract.NumericShape::new([4L]),
        storage~,
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
          @numeric_contract.ComputeDType::f32(),
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
      operations=[embedding, norm],
      final_output=@plan.ActivationRef::new(1),
      numeric_schema=schema,
    )
  catch {
    @plan.InvalidGraph(NumericActivationType) => ()
    _ => fail("unexpected activation-numeric failure")
  } noraise {
    _ => fail("mismatched activation precision was admitted")
  }
}
