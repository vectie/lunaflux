///|
fn canonical_wb_storage(
  dtype : @numeric_contract.StorageDType,
) -> @numeric_contract.TensorStorageContract {
  try! @numeric_contract.TensorStorageContract::new(
    dtype~,
    layout=@numeric_contract.StorageLayout::dense_row_major_v1(),
    encoding=@numeric_contract.StorageEncoding::plain_v1(),
    scale_granularity=@numeric_contract.ScaleGranularity::absent(),
    zero_point=@numeric_contract.ZeroPointMode::absent(),
    codebook_layout=@numeric_contract.CodebookLayout::absent(),
    conversion=@numeric_contract.NumericConversion::exact_v1(),
  )
}

///|
fn canonical_wb_schema(
  dtype? : @numeric_contract.StorageDType = @numeric_contract.StorageDType::bfloat16(),
) -> @numeric_contract.ModelNumericSchema {
  let storage = canonical_wb_storage(dtype)
  let execution = try! @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::absent(),
    activation_compute=@numeric_contract.ComputeDType::absent(),
    activation_scale=@numeric_contract.ActivationScalePolicy::absent_exact_v1(),
    tensor_input=@numeric_contract.ComputeDType::bfloat16(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::bfloat16(),
    conversion=@numeric_contract.NumericConversion::exact_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
  try! @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors=[
      @numeric_contract.NumericTensorSchema::new(
        ordinal=@numeric_contract.NumericTensorOrdinal::new(0),
        role=@numeric_contract.NumericTensorRole::parameter(),
        shape=@numeric_contract.NumericShape::new([8L, 4L]),
        storage~,
      ),
    ],
    operations=[
      @numeric_contract.NumericOperationSchema::new(
        ordinal=@numeric_contract.NumericOperationOrdinal::new(0),
        execution~,
      ),
    ],
  )
}

///|
fn canonical_wb_operation() -> PlanOperation {
  {
    id: { value: 0 },
    kind: Embedding,
    inputs: ReadOnlyArray::from_array([
      ModelInput(TokenIds),
      Tensor({ value: 0 }),
    ]),
    outputs: ReadOnlyArray::from_array([{ value: 0 }]),
    shape: EmbeddingShape(vocabulary_size=8, width=4),
    required_capability: { value: 1 },
    workspace_bytes: 0L,
    kv_layer: None,
    runtime_inputs: ReadOnlyArray::from_array([]),
    persistent_state: NoPersistentState,
  }
}

///|
fn canonical_wb_clone_operation(
  source : PlanOperation,
  id? : OperationId,
  kind? : OperationKind,
  inputs? : Array[ValueRef],
  outputs? : Array[ActivationRef],
  shape? : OperationShape,
  capability? : KernelCapabilityId,
  workspace_bytes? : Int64,
  kv_layer? : KvLayerId?,
  runtime_inputs? : Array[OperationRuntimeInput],
  persistent_state? : PersistentStateAccess,
) -> PlanOperation {
  {
    id: id.unwrap_or(source.id),
    kind: kind.unwrap_or(source.kind),
    inputs: match inputs {
      Some(values) => ReadOnlyArray::from_array(values)
      None => source.inputs
    },
    outputs: match outputs {
      Some(values) => ReadOnlyArray::from_array(values)
      None => source.outputs
    },
    shape: shape.unwrap_or(source.shape),
    required_capability: capability.unwrap_or(source.required_capability),
    workspace_bytes: workspace_bytes.unwrap_or(source.workspace_bytes),
    kv_layer: kv_layer.unwrap_or(source.kv_layer),
    runtime_inputs: match runtime_inputs {
      Some(values) => ReadOnlyArray::from_array(values)
      None => source.runtime_inputs
    },
    persistent_state: persistent_state.unwrap_or(source.persistent_state),
  }
}

///|
fn canonical_wb_constraints() -> ShapeConstraints {
  {
    max_batch_rows: 1,
    max_sequence_tokens: 8,
    vocabulary_size: 8,
    hidden_size: 4,
    intermediate_size: 8,
    attention_head_count: 2,
    kv_head_count: 1,
    head_dimension: 2,
  }
}

///|
fn canonical_wb_geometry() -> KvCacheGeometry {
  {
    layer_count: 1,
    kv_head_count: 1,
    head_dimension: 2,
    dtype: BFloat16,
    layout: LayerMajorSplitKeyValueV1,
  }
}

///|
fn canonical_wb_hash(
  constraints? : ShapeConstraints = canonical_wb_constraints(),
  geometry? : KvCacheGeometry = canonical_wb_geometry(),
  workspace? : WorkspaceBounds = { max_temporary_bytes: 0L },
  operations? : Array[PlanOperation] = [canonical_wb_operation()],
  required? : Array[KernelCapabilityId] = [{ value: 1 }],
  final_output? : ActivationRef = { value: 0 },
  mode? : ModelKvExecution = StatelessFullContext,
  schema? : @numeric_contract.ModelNumericSchema = canonical_wb_schema(),
) -> String {
  let writer = ModelPlanCanonicalWriter::new_hashing(16 * 1024 * 1024)
  try! encode_model_plan_canonical(
    writer,
    constraints,
    geometry,
    workspace,
    operations,
    required,
    final_output,
    mode,
    schema.digest(),
  )
  writer.finish_hex()
}

///|
fn assert_canonical_mutation(
  baseline : String,
  mutated : String,
  category : String,
) -> Unit raise {
  if baseline == mutated {
    fail("canonical mutation was not bound: \{category}")
  }
}

///|
fn canonical_wb_append_u64(output : Array[Byte], value : UInt64) -> Unit {
  for byte in value.to_le_bytes() {
    output.push(byte)
  }
}

///|
fn canonical_wb_append_int(output : Array[Byte], value : Int) -> Unit {
  canonical_wb_append_u64(
    output,
    Int64::from_int(value).reinterpret_as_uint64(),
  )
}

///|
fn canonical_wb_manual_minimal_bytes() -> Bytes {
  let output : Array[Byte] = []
  for byte in b"lunaflux.model-plan.v1\x00" {
    output.push(byte)
  }
  output.push(0x01)
  output.push(0x01)
  output.push(0x02)
  for value in [1, 8, 8, 4, 8, 2, 1, 2] {
    canonical_wb_append_int(output, value)
  }
  output.push(0x03)
  for value in [1, 1, 2] {
    canonical_wb_append_int(output, value)
  }
  output.push(0x01)
  output.push(0x01)
  output.push(0x04)
  canonical_wb_append_u64(output, 0UL)
  output.push(0x05)
  canonical_wb_append_int(output, 1)
  output.push(0x06)
  canonical_wb_append_int(output, 0)
  output.push(0x01)
  canonical_wb_append_int(output, 2)
  output.push(0x01)
  canonical_wb_append_int(output, 1)
  output.push(0x02)
  canonical_wb_append_int(output, 0)
  canonical_wb_append_int(output, 1)
  canonical_wb_append_int(output, 0)
  output.push(0x01)
  canonical_wb_append_int(output, 8)
  canonical_wb_append_int(output, 4)
  canonical_wb_append_int(output, 1)
  canonical_wb_append_u64(output, 0UL)
  output.push(0x00)
  canonical_wb_append_int(output, 0)
  output.push(0x00)
  output.push(0x07)
  canonical_wb_append_int(output, 1)
  canonical_wb_append_int(output, 1)
  output.push(0x08)
  canonical_wb_append_int(output, 0)
  output.push(0x09)
  let digest = canonical_wb_schema().digest().as_hex()
  for index in 0..<32 {
    let high = numeric_digest_nibble(digest[index * 2])
    let low = numeric_digest_nibble(digest[index * 2 + 1])
    output.push((high.to_int() * 16 + low.to_int()).to_byte())
  }
  Bytes::from_array(output)
}

///|
test "manual minimal oracle matches streaming canonical encoding" {
  let expected = @crypto.bytes_to_hex_string(
    @crypto.sha256(canonical_wb_manual_minimal_bytes()),
  )
  assert_eq(canonical_wb_hash(), expected)
}

///|
test "canonical encoder binds every top-level and operation category" {
  let baseline = canonical_wb_hash()
  let constraints = canonical_wb_constraints()
  let constraint_mutations : Array[ShapeConstraints] = [
    { ..constraints, max_batch_rows: 2 },
    { ..constraints, max_sequence_tokens: 9 },
    { ..constraints, vocabulary_size: 9 },
    { ..constraints, hidden_size: 5 },
    { ..constraints, intermediate_size: 9 },
    { ..constraints, attention_head_count: 3 },
    { ..constraints, kv_head_count: 2 },
    { ..constraints, head_dimension: 3 },
  ]
  for index, mutation in constraint_mutations {
    assert_canonical_mutation(
      baseline,
      canonical_wb_hash(constraints=mutation),
      "constraint-\{index}",
    )
  }
  let geometry = canonical_wb_geometry()
  let geometry_mutations : Array[KvCacheGeometry] = [
    { ..geometry, layer_count: 2 },
    { ..geometry, kv_head_count: 2 },
    { ..geometry, head_dimension: 3 },
    { ..geometry, dtype: Float16 },
  ]
  for index, mutation in geometry_mutations {
    assert_canonical_mutation(
      baseline,
      canonical_wb_hash(geometry=mutation),
      "kv-geometry-\{index}",
    )
  }
  assert_canonical_mutation(
    baseline,
    canonical_wb_hash(mode=PagedKeyValue),
    "execution-mode",
  )
  assert_canonical_mutation(
    baseline,
    canonical_wb_hash(workspace={ max_temporary_bytes: 1L }),
    "workspace",
  )
  let operation = canonical_wb_operation()
  let operation_mutations : Array[PlanOperation] = [
    canonical_wb_clone_operation(operation, id={ value: 1 }),
    canonical_wb_clone_operation(operation, kind=RmsNorm),
    canonical_wb_clone_operation(operation, inputs=[
      Tensor({ value: 0 }),
      ModelInput(TokenIds),
    ]),
    canonical_wb_clone_operation(operation, outputs=[{ value: 1 }]),
    canonical_wb_clone_operation(
      operation,
      shape=EmbeddingShape(vocabulary_size=9, width=4),
    ),
    canonical_wb_clone_operation(operation, capability={ value: 2 }),
    canonical_wb_clone_operation(operation, workspace_bytes=1L),
    canonical_wb_clone_operation(operation, kv_layer=Some({ value: 0 })),
    canonical_wb_clone_operation(operation, runtime_inputs=[StepCounts]),
    canonical_wb_clone_operation(
      operation,
      persistent_state=KvCacheReadWrite({ value: 0 }),
    ),
  ]
  for index, mutation in operation_mutations {
    assert_canonical_mutation(
      baseline,
      canonical_wb_hash(operations=[mutation]),
      "operation-\{index}",
    )
  }
  assert_canonical_mutation(
    baseline,
    canonical_wb_hash(required=[{ value: 1 }, { value: 2 }]),
    "derived-capability-order",
  )
  assert_canonical_mutation(
    baseline,
    canonical_wb_hash(final_output={ value: 1 }),
    "final-output",
  )
  assert_canonical_mutation(
    baseline,
    canonical_wb_hash(
      schema=canonical_wb_schema(dtype=@numeric_contract.StorageDType::f32()),
    ),
    "numeric-schema-digest",
  )
}

///|
test "canonical encoder preserves adjacent raw Double bit patterns" {
  let operation = canonical_wb_operation()
  let epsilon_bits = 0x3eb0c6f7a0b5ed8dUL
  let first = canonical_wb_clone_operation(
    operation,
    shape=NormalizationShape(
      width=4,
      epsilon=epsilon_bits.reinterpret_as_double(),
    ),
  )
  let second = canonical_wb_clone_operation(
    operation,
    shape=NormalizationShape(
      width=4,
      epsilon=(epsilon_bits + 1UL).reinterpret_as_double(),
    ),
  )
  assert_canonical_mutation(
    canonical_wb_hash(operations=[first]),
    canonical_wb_hash(operations=[second]),
    "raw-double-bits",
  )
}
