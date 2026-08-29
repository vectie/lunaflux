///|
fn plan_test_content() -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
}

///|
fn plan_test_constraints() -> @plan.ShapeConstraints {
  try! @plan.ShapeConstraints::new(
    max_batch_rows=1,
    max_sequence_tokens=64,
    vocabulary_size=128,
    hidden_size=16,
    intermediate_size=32,
    attention_head_count=4,
    kv_head_count=2,
    head_dimension=4,
  )
}

///|
fn plan_test_kv_geometry(
  kv_head_count? : Int = 2,
  head_dimension? : Int = 4,
) -> @plan.KvCacheGeometry {
  try! @plan.KvCacheGeometry::new(
    layer_count=1,
    kv_head_count~,
    head_dimension~,
    dtype=BFloat16,
    layout=LayerMajorSplitKeyValueV1,
  )
}

///|
fn plan_test_tensor_shape(
  operation : @plan.PlanOperation,
  input_index : Int,
) -> @numeric_contract.NumericShape {
  let dimensions : Array[Int64] = match
    (operation.kind(), operation.shape(), input_index) {
    (Embedding, EmbeddingShape(vocabulary_size~, width~), 1) =>
      [Int64::from_int(vocabulary_size), Int64::from_int(width)]
    (RmsNorm, NormalizationShape(width~, ..), 1) => [Int64::from_int(width)]
    (QkvProjection, ProjectionShape(input_width~, ..), 1) =>
      [Int64::from_int(input_width), Int64::from_int(input_width)]
    (QkvProjection, ProjectionShape(input_width~, output_width~), 2 | 3) =>
      [
        Int64::from_int((output_width - input_width) / 2),
        Int64::from_int(input_width),
      ]
    (
      OutputProjection
      | LanguageModelHead,
      ProjectionShape(input_width~, output_width~),
      1,
    ) => [Int64::from_int(output_width), Int64::from_int(input_width)]
    (GatedMlp, GatedMlpShape(hidden_size~, intermediate_size~), 1 | 2) =>
      [Int64::from_int(intermediate_size), Int64::from_int(hidden_size)]
    (GatedMlp, GatedMlpShape(hidden_size~, intermediate_size~), 3) =>
      [Int64::from_int(hidden_size), Int64::from_int(intermediate_size)]
    // Hostile role tests must reach structural validation before numeric
    // shape validation; valid tensor positions always match above.
    _ => []
  }
  try! @numeric_contract.NumericShape::new(dimensions)
}

///|
fn plan_test_numeric_schema(
  operations : ArrayView[@plan.PlanOperation],
) -> @numeric_contract.ModelNumericSchema {
  let mut tensor_count = 0
  for operation in operations {
    for input in operation.inputs() {
      if input is Tensor(reference) {
        tensor_count = tensor_count.max(reference.as_int() + 1)
      }
    }
  }
  let tensor_shapes : Array[@numeric_contract.NumericShape?] = []
  for _ in 0..<tensor_count {
    tensor_shapes.push(None)
  }
  for operation in operations {
    for input_index, input in operation.inputs() {
      match input {
        Tensor(reference) => {
          let shape = plan_test_tensor_shape(operation, input_index)
          match tensor_shapes[reference.as_int()] {
            Some(previous) =>
              if previous != shape {
                abort("plan test tensor has incompatible semantic shapes")
              }
            None => tensor_shapes[reference.as_int()] = Some(shape)
          }
        }
        Activation(_) | ModelInput(_) => ()
      }
    }
  }
  let storage = try! @numeric_contract.TensorStorageContract::new(
    dtype=@numeric_contract.StorageDType::bfloat16(),
    layout=@numeric_contract.StorageLayout::dense_row_major_v1(),
    encoding=@numeric_contract.StorageEncoding::plain_v1(),
    scale_granularity=@numeric_contract.ScaleGranularity::absent(),
    zero_point=@numeric_contract.ZeroPointMode::absent(),
    codebook_layout=@numeric_contract.CodebookLayout::absent(),
    conversion=@numeric_contract.NumericConversion::exact_v1(),
  )
  let tensors : Array[@numeric_contract.NumericTensorSchema] = []
  for index in 0..<tensor_count {
    guard tensor_shapes[index] is Some(shape) else {
      abort("plan test declared an unused tensor")
    }
    tensors.push(
      try! @numeric_contract.NumericTensorSchema::new(
        ordinal=@numeric_contract.NumericTensorOrdinal::new(index),
        role=@numeric_contract.NumericTensorRole::parameter(),
        shape~,
        storage~,
      ),
    )
  }
  let numeric_operations : Array[@numeric_contract.NumericOperationSchema] = []
  for operation in operations {
    let mut has_activation = false
    let mut has_tensor = false
    for input in operation.inputs() {
      match input {
        Activation(_) => has_activation = true
        Tensor(_) => has_tensor = true
        ModelInput(_) => ()
      }
    }
    let ordinal = try! @numeric_contract.NumericOperationOrdinal::new(
      operation.id().as_int(),
    )
    let execution = try! @numeric_contract.OperationExecutionContract::new(
      activation_input=if has_activation {
        @numeric_contract.ComputeDType::bfloat16()
      } else {
        @numeric_contract.ComputeDType::absent()
      },
      activation_compute=if has_activation {
        @numeric_contract.ComputeDType::bfloat16()
      } else {
        @numeric_contract.ComputeDType::absent()
      },
      activation_scale=@numeric_contract.ActivationScalePolicy::absent_exact_v1(),
      tensor_input=if has_tensor {
        @numeric_contract.ComputeDType::bfloat16()
      } else {
        @numeric_contract.ComputeDType::absent()
      },
      accumulator=@numeric_contract.AccumulatorDType::f32(),
      output=@numeric_contract.ComputeDType::bfloat16(),
      conversion=@numeric_contract.NumericConversion::exact_v1(),
      order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
    )
    numeric_operations.push(
      @numeric_contract.NumericOperationSchema::new(ordinal~, execution~),
    )
  }
  try! @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors~,
    operations=numeric_operations,
  )
}

///|
test "KV layer identities are nonnegative semantic scalars" {
  assert_eq(try! @plan.KvLayerId::new(3).as_int(), 3)
  try @plan.KvLayerId::new(-1) catch {
    InvalidIdentifier(KvLayer) => ()
    _ => fail("unexpected KV layer identity failure")
  } noraise {
    _ => fail("expected negative KV layer rejection")
  }
}

///|
test "stateless operation construction rejects KV-bearing operation kinds" {
  let kv_kinds : Array[@plan.OperationKind] = [RotaryEmbedding, CausalAttention]
  for kind in kv_kinds {
    try
      @plan.PlanOperation::new_stateless(
        id=@plan.OperationId::new(0),
        kind~,
        inputs=[Activation(@plan.ActivationRef::new(0))],
        outputs=[@plan.ActivationRef::new(1)],
        shape=AttentionShape(query_heads=4, kv_heads=2, head_dimension=4),
        required_capability=@plan.KernelCapabilityId::causal_attention(),
        workspace_bytes=0L,
      )
    catch {
      InvalidGraph(KvExecutionContract) => ()
      _ => fail("unexpected stateless KV operation failure")
    } noraise {
      _ => fail("expected stateless KV operation rejection")
    }
  }
}

///|
test "live non-KV operation construction requires an exact nonempty contract" {
  let live = @plan.PlanOperation::new_with_runtime_inputs(
    id=@plan.OperationId::new(0),
    kind=Embedding,
    inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
    outputs=[@plan.ActivationRef::new(0)],
    shape=EmbeddingShape(vocabulary_size=128, width=16),
    required_capability=@plan.KernelCapabilityId::embedding_lookup(),
    workspace_bytes=0L,
    runtime_inputs=[StepCounts],
  )
  assert_eq(live.runtime_inputs(), [StepCounts])
  assert_eq(live.kv_layer(), None)
  assert_eq(live.persistent_state(), NoPersistentState)
  try
    @plan.PlanOperation::new_with_runtime_inputs(
      id=@plan.OperationId::new(0),
      kind=Embedding,
      inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
      outputs=[@plan.ActivationRef::new(0)],
      shape=EmbeddingShape(vocabulary_size=128, width=16),
      required_capability=@plan.KernelCapabilityId::embedding_lookup(),
      workspace_bytes=0L,
      runtime_inputs=[],
    )
  catch {
    InvalidGraph(KvExecutionContract) => ()
    _ => fail("unexpected empty live-input failure")
  } noraise {
    _ => fail("expected empty live-input rejection")
  }
}

///|
test "stateless plan construction rejects explicitly layered attention" {
  let attention = @plan.PlanOperation::new_with_execution(
    id=@plan.OperationId::new(0),
    kind=CausalAttention,
    inputs=[Activation(@plan.ActivationRef::new(0))],
    outputs=[@plan.ActivationRef::new(1)],
    shape=AttentionShape(query_heads=4, kv_heads=2, head_dimension=4),
    required_capability=@plan.KernelCapabilityId::causal_attention(),
    workspace_bytes=0L,
    kv_layer=@plan.KvLayerId::new(0),
    runtime_inputs=[],
    persistent_state=NoPersistentState,
  )
  try
    @plan.ModelPlan::new_stateless(
      content_digest=plan_test_content(),
      limits=@plan.ModelPlanLimits::production(),
      shape_constraints=plan_test_constraints(),
      kv_cache_geometry=plan_test_kv_geometry(),
      workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
      operations=[attention],
      final_output=@plan.ActivationRef::new(1),
      numeric_schema=plan_test_numeric_schema([attention]),
    )
  catch {
    InvalidGraph(KvExecutionContract) => ()
    _ => fail("unexpected stateless plan KV failure")
  } noraise {
    _ => fail("expected stateless plan KV rejection")
  }
}

///|
fn embedding_operation(
  first_input : @plan.ValueRef,
  capability : @plan.KernelCapabilityId,
) -> @plan.PlanOperation raise {
  @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(0),
    kind=Embedding,
    inputs=[first_input, Tensor(@plan.TensorRef::new(0))],
    outputs=[@plan.ActivationRef::new(0)],
    shape=EmbeddingShape(vocabulary_size=128, width=16),
    required_capability=capability,
    workspace_bytes=0L,
  )
}

///|
test "model plan copies construction arrays into immutable ordered storage" {
  let operation = embedding_operation(
    ModelInput(TokenIds),
    @plan.KernelCapabilityId::embedding_lookup(),
  )
  let operations = [operation]
  let model_plan = @plan.ModelPlan::new_stateless(
    content_digest=plan_test_content(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=plan_test_constraints(),
    kv_cache_geometry=plan_test_kv_geometry(),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=4096L),
    operations~,
    final_output=@plan.ActivationRef::new(0),
    numeric_schema=plan_test_numeric_schema(operations),
  )
  operations.push(operation)
  assert_eq(model_plan.operations().length(), 1)
  assert_eq(model_plan.required_capabilities().length(), 1)
  assert_eq(model_plan.final_output().as_int(), 0)
  assert_eq(model_plan.kv_cache_geometry(), plan_test_kv_geometry())
}

///|
test "model plan rejects KV geometry inconsistent with semantic dimensions" {
  let operation = embedding_operation(
    ModelInput(TokenIds),
    @plan.KernelCapabilityId::embedding_lookup(),
  )
  try
    @plan.ModelPlan::new_stateless(
      content_digest=plan_test_content(),
      limits=@plan.ModelPlanLimits::production(),
      shape_constraints=plan_test_constraints(),
      kv_cache_geometry=plan_test_kv_geometry(kv_head_count=1),
      workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=4096L),
      operations=[operation],
      final_output=@plan.ActivationRef::new(0),
      numeric_schema=plan_test_numeric_schema([operation]),
    )
  catch {
    InvalidShape(KvCacheShapeMismatch) => ()
    _ => fail("unexpected KV shape-consistency error")
  } noraise {
    _ => fail("expected inconsistent KV geometry rejection")
  }
}

///|
test "query width is independent of hidden width and remains overflow checked" {
  let constraints = @plan.ShapeConstraints::new(
    max_batch_rows=1,
    max_sequence_tokens=64,
    vocabulary_size=128,
    hidden_size=17,
    intermediate_size=32,
    attention_head_count=4,
    kv_head_count=2,
    head_dimension=4,
  )
  assert_eq(constraints.hidden_size(), 17)
  assert_eq(constraints.query_width(), 16)
  assert_eq(constraints.kv_width(), 8)
  try
    @plan.ShapeConstraints::new(
      max_batch_rows=1,
      max_sequence_tokens=64,
      vocabulary_size=128,
      hidden_size=17,
      intermediate_size=32,
      attention_head_count=2147483647,
      kv_head_count=1,
      head_dimension=2,
    )
  catch {
    InvalidShape(DimensionOverflow) => ()
    _ => fail("unexpected validation error")
  } noraise {
    _ => fail("expected overflowing query width rejection")
  }
}

///|
test "model plan rejects forward activation references" {
  let operation = @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(0),
    kind=RmsNorm,
    inputs=[
      Activation(@plan.ActivationRef::new(9)),
      Tensor(@plan.TensorRef::new(0)),
    ],
    outputs=[@plan.ActivationRef::new(0)],
    shape=NormalizationShape(width=16, epsilon=0.000001),
    required_capability=@plan.KernelCapabilityId::rms_norm(),
    workspace_bytes=0L,
  )
  try
    @plan.ModelPlan::new_stateless(
      content_digest=plan_test_content(),
      limits=@plan.ModelPlanLimits::production(),
      shape_constraints=plan_test_constraints(),
      kv_cache_geometry=plan_test_kv_geometry(),
      workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=4096L),
      operations=[operation],
      final_output=@plan.ActivationRef::new(0),
      numeric_schema=plan_test_numeric_schema([operation]),
    )
  catch {
    InvalidGraph(MissingActivationInput) => ()
    _ => fail("unexpected validation error")
  } noraise {
    _ => fail("expected missing activation input")
  }
}

///|
test "operation kind requires its stable semantic capability" {
  let operation = embedding_operation(
    ModelInput(TokenIds),
    @plan.KernelCapabilityId::rms_norm(),
  )
  try
    @plan.ModelPlan::new_stateless(
      content_digest=plan_test_content(),
      limits=@plan.ModelPlanLimits::production(),
      shape_constraints=plan_test_constraints(),
      kv_cache_geometry=plan_test_kv_geometry(),
      workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=4096L),
      operations=[operation],
      final_output=@plan.ActivationRef::new(0),
      numeric_schema=plan_test_numeric_schema([operation]),
    )
  catch {
    InvalidGraph(OperationCapabilityMismatch) => ()
    _ => fail("unexpected validation error")
  } noraise {
    _ => fail("expected operation capability mismatch")
  }
}

///|
test "operation kinds enforce exact semantic input roles" {
  let operation = embedding_operation(
    Tensor(@plan.TensorRef::new(1)),
    @plan.KernelCapabilityId::embedding_lookup(),
  )
  try
    @plan.ModelPlan::new_stateless(
      content_digest=plan_test_content(),
      limits=@plan.ModelPlanLimits::production(),
      shape_constraints=plan_test_constraints(),
      kv_cache_geometry=plan_test_kv_geometry(),
      workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=4096L),
      operations=[operation],
      final_output=@plan.ActivationRef::new(0),
      numeric_schema=plan_test_numeric_schema([operation]),
    )
  catch {
    InvalidGraph(OperationInputRoles) => ()
    _ => fail("unexpected validation error")
  } noraise {
    _ => fail("expected operation input-role mismatch")
  }
}
