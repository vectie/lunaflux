///|
fn i8_fixture_content() -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
}

///|
fn i8_fixture_metadata() -> @spec.LlamaModelMetadata {
  let spec = try! @spec.LlamaModelSpec::new(
    vocabulary_size=8,
    hidden_size=4,
    intermediate_size=8,
    layer_count=1,
    attention_head_count=2,
    kv_head_count=1,
    context_length=8,
    rope_theta=10000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  try! @spec.LlamaModelMetadata::from_verified_content(
    spec,
    i8_fixture_content(),
  )
}

///|
fn i8_fixture_constraints() -> @plan.ShapeConstraints {
  try! @plan.ShapeConstraints::new(
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
fn i8_fixture_geometry() -> @plan.KvCacheGeometry {
  try! @plan.KvCacheGeometry::new(
    layer_count=1,
    kv_head_count=1,
    head_dimension=2,
    dtype=BFloat16,
    layout=LayerMajorSplitKeyValueV1,
  )
}

///|
fn i8_fixture_operations() -> Array[@plan.PlanOperation] {
  [
    try! @plan.PlanOperation::new_stateless(
      id=@plan.OperationId::new(0),
      kind=Embedding,
      inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
      outputs=[@plan.ActivationRef::new(0)],
      shape=EmbeddingShape(vocabulary_size=8, width=4),
      required_capability=@plan.KernelCapabilityId::embedding_lookup(),
      workspace_bytes=0L,
    ),
    try! @plan.PlanOperation::new_stateless(
      id=@plan.OperationId::new(1),
      kind=GatedMlp,
      inputs=[
        Activation(@plan.ActivationRef::new(0)),
        Tensor(@plan.TensorRef::new(1)),
        Tensor(@plan.TensorRef::new(2)),
        Tensor(@plan.TensorRef::new(3)),
      ],
      outputs=[@plan.ActivationRef::new(1)],
      shape=GatedMlpShape(hidden_size=4, intermediate_size=8),
      required_capability=@plan.KernelCapabilityId::gated_mlp(),
      workspace_bytes=0L,
    ),
  ]
}

///|
fn i8_fixture_qkv_operations() -> Array[@plan.PlanOperation] {
  let operations = i8_fixture_operations()
  operations[1] = try! @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(1),
    kind=QkvProjection,
    inputs=[
      Activation(@plan.ActivationRef::new(0)),
      Tensor(@plan.TensorRef::new(1)),
      Tensor(@plan.TensorRef::new(2)),
      Tensor(@plan.TensorRef::new(3)),
    ],
    outputs=[@plan.ActivationRef::new(1)],
    shape=ProjectionShape(input_width=4, output_width=8),
    required_capability=@plan.KernelCapabilityId::qkv_projection(),
    workspace_bytes=0L,
  )
  operations
}

///|
fn i8_fixture_plain_bf16() -> @numeric_contract.TensorStorageContract {
  try! @numeric_contract.TensorStorageContract::new(
    dtype=@numeric_contract.StorageDType::bfloat16(),
    layout=@numeric_contract.StorageLayout::dense_row_major_v1(),
    encoding=@numeric_contract.StorageEncoding::plain_v1(),
    scale_granularity=@numeric_contract.ScaleGranularity::absent(),
    zero_point=@numeric_contract.ZeroPointMode::absent(),
    codebook_layout=@numeric_contract.CodebookLayout::absent(),
    conversion=@numeric_contract.NumericConversion::exact_v1(),
  )
}

///|
fn i8_fixture_execution(
  i8 : Bool,
  activation : Bool,
) -> @numeric_contract.OperationExecutionContract {
  if i8 {
    @numeric_contract.OperationExecutionContract::symmetric_i8_weight_only_v1()
  } else {
    try! @numeric_contract.OperationExecutionContract::new(
      activation_input=if activation {
        @numeric_contract.ComputeDType::bfloat16()
      } else {
        @numeric_contract.ComputeDType::absent()
      },
      activation_compute=if activation {
        @numeric_contract.ComputeDType::bfloat16()
      } else {
        @numeric_contract.ComputeDType::absent()
      },
      activation_scale=@numeric_contract.ActivationScalePolicy::absent_exact_v1(),
      tensor_input=@numeric_contract.ComputeDType::bfloat16(),
      accumulator=@numeric_contract.AccumulatorDType::f32(),
      output=@numeric_contract.ComputeDType::bfloat16(),
      conversion=@numeric_contract.NumericConversion::exact_v1(),
      order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
    )
  }
}

///|
fn i8_fixture_manual_schema(
  modes : ArrayView[Bool],
  orphan_scale : Bool,
) -> @numeric_contract.ModelNumericSchema {
  let shapes : Array[Array[Int64]] = [[8L, 4L], [8L, 4L], [8L, 4L], [4L, 8L]]
  let tensors : Array[@numeric_contract.NumericTensorSchema] = []
  let mut scale_count = 0
  for index in 0..<4 {
    if modes[index] {
      tensors.push(
        try! @numeric_contract.NumericTensorSchema::new(
          ordinal=@numeric_contract.NumericTensorOrdinal::new(index),
          role=@numeric_contract.NumericTensorRole::parameter(),
          shape=@numeric_contract.NumericShape::new(shapes[index]),
          storage=@numeric_contract.TensorStorageContract::symmetric_i8_weight_only_v1(),
          scale=@numeric_contract.ScaleTensorOrdinal::new(
            @numeric_contract.NumericTensorOrdinal::new(4 + scale_count),
          ),
        ),
      )
      scale_count += 1
    } else {
      tensors.push(
        try! @numeric_contract.NumericTensorSchema::new(
          ordinal=@numeric_contract.NumericTensorOrdinal::new(index),
          role=@numeric_contract.NumericTensorRole::parameter(),
          shape=@numeric_contract.NumericShape::new(shapes[index]),
          storage=i8_fixture_plain_bf16(),
        ),
      )
    }
  }
  let mut selected_scale = 0
  for index in 0..<4 {
    if modes[index] {
      tensors.push(
        try! @numeric_contract.NumericTensorSchema::new(
          ordinal=@numeric_contract.NumericTensorOrdinal::new(
            4 + selected_scale,
          ),
          role=@numeric_contract.NumericTensorRole::scale_metadata(),
          shape=@numeric_contract.NumericShape::new([shapes[index][0]]),
          storage=@numeric_contract.TensorStorageContract::symmetric_i8_weight_only_scale_v1(),
        ),
      )
      selected_scale += 1
    }
  }
  if orphan_scale {
    tensors.push(
      try! @numeric_contract.NumericTensorSchema::new(
        ordinal=@numeric_contract.NumericTensorOrdinal::new(tensors.length()),
        role=@numeric_contract.NumericTensorRole::scale_metadata(),
        shape=@numeric_contract.NumericShape::new([1L]),
        storage=@numeric_contract.TensorStorageContract::symmetric_i8_weight_only_scale_v1(),
      ),
    )
  }
  try! @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors~,
    operations=[
      @numeric_contract.NumericOperationSchema::new(
        ordinal=@numeric_contract.NumericOperationOrdinal::new(0),
        execution=i8_fixture_execution(false, false),
      ),
      @numeric_contract.NumericOperationSchema::new(
        ordinal=@numeric_contract.NumericOperationOrdinal::new(1),
        execution=i8_fixture_execution(true, true),
      ),
    ],
  )
}

///|
fn i8_fixture_partial_qkv_schema() -> @numeric_contract.ModelNumericSchema {
  let bf16 = i8_fixture_plain_bf16()
  let i8 = @numeric_contract.TensorStorageContract::symmetric_i8_weight_only_v1()
  let scale = @numeric_contract.TensorStorageContract::symmetric_i8_weight_only_scale_v1()
  try! @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors=[
      @numeric_contract.NumericTensorSchema::new(
        ordinal=@numeric_contract.NumericTensorOrdinal::new(0),
        role=@numeric_contract.NumericTensorRole::parameter(),
        shape=@numeric_contract.NumericShape::new([8L, 4L]),
        storage=bf16,
      ),
      @numeric_contract.NumericTensorSchema::new(
        ordinal=@numeric_contract.NumericTensorOrdinal::new(1),
        role=@numeric_contract.NumericTensorRole::parameter(),
        shape=@numeric_contract.NumericShape::new([4L, 4L]),
        storage=i8,
        scale=@numeric_contract.ScaleTensorOrdinal::new(
          @numeric_contract.NumericTensorOrdinal::new(4),
        ),
      ),
      @numeric_contract.NumericTensorSchema::new(
        ordinal=@numeric_contract.NumericTensorOrdinal::new(2),
        role=@numeric_contract.NumericTensorRole::parameter(),
        shape=@numeric_contract.NumericShape::new([2L, 4L]),
        storage=bf16,
      ),
      @numeric_contract.NumericTensorSchema::new(
        ordinal=@numeric_contract.NumericTensorOrdinal::new(3),
        role=@numeric_contract.NumericTensorRole::parameter(),
        shape=@numeric_contract.NumericShape::new([2L, 4L]),
        storage=i8,
        scale=@numeric_contract.ScaleTensorOrdinal::new(
          @numeric_contract.NumericTensorOrdinal::new(5),
        ),
      ),
      @numeric_contract.NumericTensorSchema::new(
        ordinal=@numeric_contract.NumericTensorOrdinal::new(4),
        role=@numeric_contract.NumericTensorRole::scale_metadata(),
        shape=@numeric_contract.NumericShape::new([4L]),
        storage=scale,
      ),
      @numeric_contract.NumericTensorSchema::new(
        ordinal=@numeric_contract.NumericTensorOrdinal::new(5),
        role=@numeric_contract.NumericTensorRole::scale_metadata(),
        shape=@numeric_contract.NumericShape::new([2L]),
        storage=scale,
      ),
    ],
    operations=[
      @numeric_contract.NumericOperationSchema::new(
        ordinal=@numeric_contract.NumericOperationOrdinal::new(0),
        execution=i8_fixture_execution(false, false),
      ),
      @numeric_contract.NumericOperationSchema::new(
        ordinal=@numeric_contract.NumericOperationOrdinal::new(1),
        execution=i8_fixture_execution(true, true),
      ),
    ],
  )
}

///|
fn i8_fixture_plan(
  operations : ArrayView[@plan.PlanOperation],
  schema : @numeric_contract.ModelNumericSchema,
  content_digest? : @spec.ContentDigest = i8_fixture_content(),
) -> @plan.ModelPlan raise {
  @plan.ModelPlan::new_stateless(
    content_digest~,
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=i8_fixture_constraints(),
    kv_cache_geometry=i8_fixture_geometry(),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations~,
    final_output=@plan.ActivationRef::new(1),
    numeric_schema=schema,
  )
}

///|
test "one content-only metadata admits distinct BF16 FP8 and I8 plan identities" {
  let metadata = i8_fixture_metadata()
  let operations = i8_fixture_operations()
  let selected = [@plan.OperationId::new(1)]
  let bf16 = i8_fixture_plan(
    operations,
    synthetic_bf16_numeric_schema(operations),
    content_digest=metadata.content_digest(),
  )
  let fp8 = i8_fixture_plan(
    operations,
    synthetic_fp8_w8a8_numeric_schema(operations, selected),
    content_digest=metadata.content_digest(),
  )
  let i8 = i8_fixture_plan(
    operations,
    synthetic_symmetric_i8_weight_only_numeric_schema_v1(operations, selected),
    content_digest=metadata.content_digest(),
  )
  assert_false(bf16.identity().plan() == fp8.identity().plan())
  assert_false(bf16.identity().plan() == i8.identity().plan())
  assert_false(fp8.identity().plan() == i8.identity().plan())
  assert_eq(bf16.identity().content(), fp8.identity().content())
  assert_eq(fp8.identity().content(), i8.identity().content())
}

///|
test "I8 fixture atomically binds every selected matrix and exact scale ref" {
  let operations = i8_fixture_operations()
  let schema = synthetic_symmetric_i8_weight_only_numeric_schema_v1(
    operations,
    [@plan.OperationId::new(1)],
  )
  let plan = i8_fixture_plan(operations, schema)
  assert_eq(plan.tensor_count(), 7)
  for index in 1..<4 {
    guard plan.tensor(@plan.TensorRef::new(index)) is Some(tensor) &&
      tensor.scale() is Some(scale) else {
      fail("selected I8 weight lost its exact scale reference")
    }
    assert_eq(scale.tensor(), @plan.TensorRef::new(index + 3))
  }
}

///|
test "I8 operation rejects partial multi-weight quantization" {
  let schema = i8_fixture_manual_schema([false, true, false, true], false)
  try i8_fixture_plan(i8_fixture_operations(), schema) catch {
    @plan.InvalidGraph(NumericTensorType) => ()
    _ => fail("unexpected partial-I8 binding failure")
  } noraise {
    _ => fail("partially quantized multi-weight operation was admitted")
  }
}

///|
test "I8 QKV operation rejects partial Q K V quantization" {
  try
    i8_fixture_plan(
      i8_fixture_qkv_operations(),
      i8_fixture_partial_qkv_schema(),
    )
  catch {
    @plan.InvalidGraph(NumericTensorType) => ()
    _ => fail("unexpected partial-I8 QKV binding failure")
  } noraise {
    _ => fail("partially quantized QKV operation was admitted")
  }
}

///|
test "I8 plan rejects orphan scale metadata" {
  let schema = i8_fixture_manual_schema([false, true, true, true], true)
  try i8_fixture_plan(i8_fixture_operations(), schema) catch {
    @plan.InvalidGraph(UnreferencedNumericMetadata) => ()
    _ => fail("unexpected orphan-I8-scale failure")
  } noraise {
    _ => fail("orphan I8 scale metadata was admitted")
  }
}

///|
test "panic I8 fixture rejects shared BF16 and I8 weight uses" {
  let operations = i8_fixture_operations()
  let mlp = try! @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(1),
    kind=GatedMlp,
    inputs=[
      Activation(@plan.ActivationRef::new(0)),
      Tensor(@plan.TensorRef::new(0)),
      Tensor(@plan.TensorRef::new(1)),
      Tensor(@plan.TensorRef::new(2)),
    ],
    outputs=[@plan.ActivationRef::new(1)],
    shape=GatedMlpShape(hidden_size=4, intermediate_size=8),
    required_capability=@plan.KernelCapabilityId::gated_mlp(),
    workspace_bytes=0L,
  )
  operations[1] = mlp
  ignore(
    synthetic_symmetric_i8_weight_only_numeric_schema_v1(operations, [
      @plan.OperationId::new(1),
    ]),
  )
}

///|
test "panic I8 v1 fixture rejects unknown selected operation IDs" {
  ignore(
    synthetic_symmetric_i8_weight_only_numeric_schema_v1(
      i8_fixture_operations(),
      [@plan.OperationId::new(99)],
    ),
  )
}

///|
test "panic I8 v1 fixture rejects duplicate requested operation IDs" {
  ignore(
    synthetic_symmetric_i8_weight_only_numeric_schema_v1(
      i8_fixture_operations(),
      [@plan.OperationId::new(1), @plan.OperationId::new(1)],
    ),
  )
}

///|
test "panic I8 v1 fixture rejects selected IDs present twice in the graph" {
  let operations = i8_fixture_operations()
  operations.push(operations[1])
  ignore(
    synthetic_symmetric_i8_weight_only_numeric_schema_v1(operations, [
      @plan.OperationId::new(1),
    ]),
  )
}

///|
test "panic I8 v1 fixture rejects future-shaped unreviewed operation kinds" {
  let operations = i8_fixture_operations()
  operations[1] = try! @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(1),
    kind=Embedding,
    inputs=[
      Activation(@plan.ActivationRef::new(0)),
      Tensor(@plan.TensorRef::new(1)),
    ],
    outputs=[@plan.ActivationRef::new(1)],
    shape=EmbeddingShape(vocabulary_size=8, width=4),
    required_capability=@plan.KernelCapabilityId::embedding_lookup(),
    workspace_bytes=0L,
  )
  ignore(
    synthetic_symmetric_i8_weight_only_numeric_schema_v1(operations, [
      @plan.OperationId::new(1),
    ]),
  )
}
