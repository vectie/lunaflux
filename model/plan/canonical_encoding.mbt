///|
/// Streaming canonical-plan encoder. `hasher=None` performs the exact byte
/// preflight without retaining canonical bytes; `Some` hashes the same walk.
priv struct ModelPlanCanonicalWriter {
  hasher : @canonical_sha256.CanonicalSha256?
  max_count : Int
  mut count : Int
}

///|
fn ModelPlanCanonicalWriter::new_counting(
  max_count : Int,
) -> ModelPlanCanonicalWriter {
  { hasher: None, max_count, count: 0 }
}

///|
fn ModelPlanCanonicalWriter::new_hashing(
  max_count : Int,
) -> ModelPlanCanonicalWriter {
  {
    hasher: Some(@canonical_sha256.CanonicalSha256::new()),
    max_count,
    count: 0,
  }
}

///|
fn ModelPlanCanonicalWriter::reserve(
  self : ModelPlanCanonicalWriter,
  amount : Int,
) -> Unit raise PlanValidationError {
  if amount < 0 || self.count > self.max_count - amount {
    raise LimitExceeded(CanonicalBytes)
  }
  self.count += amount
}

///|
fn ModelPlanCanonicalWriter::write_bytes(
  self : ModelPlanCanonicalWriter,
  bytes : BytesView,
) -> Unit raise PlanValidationError {
  self.reserve(bytes.length())
  match self.hasher {
    Some(hasher) =>
      if !hasher.update_bytes(bytes) {
        abort("finalized plan canonical hash")
      }
    None => ()
  }
}

///|
fn ModelPlanCanonicalWriter::write_byte(
  self : ModelPlanCanonicalWriter,
  byte : Byte,
) -> Unit raise PlanValidationError {
  self.reserve(1)
  match self.hasher {
    Some(hasher) =>
      if !hasher.update_byte(byte) {
        abort("finalized plan canonical hash")
      }
    None => ()
  }
}

///|
fn ModelPlanCanonicalWriter::write_u64(
  self : ModelPlanCanonicalWriter,
  value : UInt64,
) -> Unit raise PlanValidationError {
  self.reserve(8)
  match self.hasher {
    Some(hasher) =>
      if !hasher.update_u64_le(value) {
        abort("finalized plan canonical hash")
      }
    None => ()
  }
}

///|
fn ModelPlanCanonicalWriter::write_int(
  self : ModelPlanCanonicalWriter,
  value : Int,
) -> Unit raise PlanValidationError {
  self.write_u64(Int64::from_int(value).reinterpret_as_uint64())
}

///|
fn ModelPlanCanonicalWriter::write_int64(
  self : ModelPlanCanonicalWriter,
  value : Int64,
) -> Unit raise PlanValidationError {
  self.write_u64(value.reinterpret_as_uint64())
}

///|
fn ModelPlanCanonicalWriter::finish_hex(
  self : ModelPlanCanonicalWriter,
) -> String {
  guard self.hasher is Some(hasher) else {
    abort("counting writer has no hash")
  }
  hasher.finalize_hex()
}

///|
fn encode_execution_mode(
  writer : ModelPlanCanonicalWriter,
  mode : ModelKvExecution,
) -> Unit raise PlanValidationError {
  writer.write_byte(0x01)
  writer.write_byte(execution_mode_tag(mode))
}

///|
fn encode_shape_constraints(
  writer : ModelPlanCanonicalWriter,
  constraints : ShapeConstraints,
) -> Unit raise PlanValidationError {
  writer.write_byte(0x02)
  writer.write_int(constraints.max_batch_rows())
  writer.write_int(constraints.max_sequence_tokens())
  writer.write_int(constraints.vocabulary_size())
  writer.write_int(constraints.hidden_size())
  writer.write_int(constraints.intermediate_size())
  writer.write_int(constraints.attention_head_count())
  writer.write_int(constraints.kv_head_count())
  writer.write_int(constraints.head_dimension())
}

///|
fn encode_kv_geometry(
  writer : ModelPlanCanonicalWriter,
  geometry : KvCacheGeometry,
) -> Unit raise PlanValidationError {
  writer.write_byte(0x03)
  writer.write_int(geometry.layer_count())
  writer.write_int(geometry.kv_head_count())
  writer.write_int(geometry.head_dimension())
  writer.write_byte(model_dtype_tag(geometry.dtype()))
  writer.write_byte(kv_cache_layout_tag(geometry.layout()))
}

///|
fn encode_value_ref(
  writer : ModelPlanCanonicalWriter,
  value : ValueRef,
) -> Unit raise PlanValidationError {
  writer.write_byte(value_ref_tag(value))
  match value {
    ModelInput(TokenIds) => writer.write_int(1)
    Tensor(reference) => writer.write_int(reference.as_int())
    Activation(reference) => writer.write_int(reference.as_int())
  }
}

///|
fn encode_operation_shape(
  writer : ModelPlanCanonicalWriter,
  shape : OperationShape,
) -> Unit raise PlanValidationError {
  writer.write_byte(operation_shape_tag(shape))
  match shape {
    EmbeddingShape(vocabulary_size~, width~) => {
      writer.write_int(vocabulary_size)
      writer.write_int(width)
    }
    NormalizationShape(width~, epsilon~) => {
      writer.write_int(width)
      writer.write_u64(epsilon.reinterpret_as_uint64())
    }
    ProjectionShape(input_width~, output_width~) => {
      writer.write_int(input_width)
      writer.write_int(output_width)
    }
    RotaryShape(query_heads~, kv_heads~, head_dimension~, rope_theta~) => {
      writer.write_int(query_heads)
      writer.write_int(kv_heads)
      writer.write_int(head_dimension)
      writer.write_u64(rope_theta.reinterpret_as_uint64())
    }
    AttentionShape(query_heads~, kv_heads~, head_dimension~) => {
      writer.write_int(query_heads)
      writer.write_int(kv_heads)
      writer.write_int(head_dimension)
    }
    ElementwiseShape(width~) => writer.write_int(width)
    GatedMlpShape(hidden_size~, intermediate_size~) => {
      writer.write_int(hidden_size)
      writer.write_int(intermediate_size)
    }
  }
}

///|
fn encode_operation(
  writer : ModelPlanCanonicalWriter,
  operation : PlanOperation,
) -> Unit raise PlanValidationError {
  writer.write_byte(0x06)
  writer.write_int(operation.id().as_int())
  writer.write_byte(operation_kind_tag(operation.kind()))
  writer.write_int(operation.inputs().length())
  for input in operation.inputs() {
    encode_value_ref(writer, input)
  }
  writer.write_int(operation.outputs().length())
  for output in operation.outputs() {
    writer.write_int(output.as_int())
  }
  encode_operation_shape(writer, operation.shape())
  writer.write_int(operation.required_capability().as_int())
  writer.write_int64(operation.workspace_bytes())
  writer.write_byte(optional_kv_layer_tag(operation.kv_layer()))
  match operation.kv_layer() {
    Some(layer) => writer.write_int(layer.as_int())
    None => ()
  }
  writer.write_int(operation.runtime_inputs().length())
  for input in operation.runtime_inputs() {
    writer.write_byte(runtime_input_tag(input))
  }
  writer.write_byte(persistent_state_tag(operation.persistent_state()))
  match operation.persistent_state() {
    NoPersistentState => ()
    KvCacheReadWrite(layer) => writer.write_int(layer.as_int())
  }
}

///|
fn numeric_digest_nibble(value : UInt16) -> Byte {
  let zero : UInt16 = 48
  let nine : UInt16 = 57
  let lower_a : UInt16 = 97
  if zero <= value && value <= nine {
    (value.to_int() - zero.to_int()).to_byte()
  } else {
    (value.to_int() - lower_a.to_int() + 10).to_byte()
  }
}

///|
fn encode_numeric_schema_digest(
  writer : ModelPlanCanonicalWriter,
  digest : @numeric_contract.ModelNumericSchemaDigest,
) -> Unit raise PlanValidationError {
  let hex = digest.as_hex()
  if hex.length() != 64 ||
    !hex.all(c => c.is_ascii_digit() || ('a' <= c && c <= 'f')) {
    abort("numeric schema produced an invalid SHA-256 digest")
  }
  writer.write_byte(0x09)
  for index in 0..<32 {
    let high = numeric_digest_nibble(hex[index * 2])
    let low = numeric_digest_nibble(hex[index * 2 + 1])
    writer.write_byte((high.to_int() * 16 + low.to_int()).to_byte())
  }
}

///|
fn encode_model_plan_canonical(
  writer : ModelPlanCanonicalWriter,
  shape_constraints : ShapeConstraints,
  kv_cache_geometry : KvCacheGeometry,
  workspace_bounds : WorkspaceBounds,
  operations : ArrayView[PlanOperation],
  required_capabilities : ArrayView[KernelCapabilityId],
  final_output : ActivationRef,
  kv_execution : ModelKvExecution,
  numeric_schema_digest : @numeric_contract.ModelNumericSchemaDigest,
) -> Unit raise PlanValidationError {
  writer.write_bytes(b"lunaflux.model-plan.v1\x00")
  encode_execution_mode(writer, kv_execution)
  encode_shape_constraints(writer, shape_constraints)
  encode_kv_geometry(writer, kv_cache_geometry)
  writer.write_byte(0x04)
  writer.write_int64(workspace_bounds.max_temporary_bytes())
  writer.write_byte(0x05)
  writer.write_int(operations.length())
  for operation in operations {
    encode_operation(writer, operation)
  }
  writer.write_byte(0x07)
  writer.write_int(required_capabilities.length())
  for capability in required_capabilities {
    writer.write_int(capability.as_int())
  }
  writer.write_byte(0x08)
  writer.write_int(final_output.as_int())
  encode_numeric_schema_digest(writer, numeric_schema_digest)
}

///|
fn count_required_capabilities(operations : ArrayView[PlanOperation]) -> Int {
  let seen = FixedArray::make(11, false)
  let mut count = 0
  for operation in operations {
    let value = expected_capability(operation).as_int()
    if !seen[value] {
      seen[value] = true
      count += 1
    }
  }
  count
}

///|
fn preflight_model_plan(
  limits : ModelPlanLimits,
  shape_constraints : ShapeConstraints,
  kv_cache_geometry : KvCacheGeometry,
  workspace_bounds : WorkspaceBounds,
  operations : ArrayView[PlanOperation],
  final_output : ActivationRef,
  kv_execution : ModelKvExecution,
  numeric_schema : @numeric_contract.ModelNumericSchema,
) -> Int raise PlanValidationError {
  limits.check_operation_count(operations.length())
  limits.check_numeric_tensor_count(numeric_schema.tensors().length())
  if numeric_schema.operations().length() != operations.length() {
    raise InvalidGraph(NumericOperationBinding)
  }
  let mut total_inputs = 0
  for operation in operations {
    limits.check_outputs_per_operation(operation.outputs().length())
    limits.check_value_inputs_per_operation(operation.inputs().length())
    limits.check_runtime_inputs_per_operation(
      operation.runtime_inputs().length(),
    )
    total_inputs += operation.inputs().length()
    limits.check_total_value_inputs(total_inputs)
  }
  let required_count = count_required_capabilities(operations)
  let required = FixedArray::make(
    required_count,
    KernelCapabilityId::embedding_lookup(),
  )
  let seen = FixedArray::make(11, false)
  let mut next_required = 0
  for operation in operations {
    let capability = expected_capability(operation)
    let value = capability.as_int()
    if !seen[value] {
      seen[value] = true
      required[next_required] = capability
      next_required += 1
    }
  }
  let counter = ModelPlanCanonicalWriter::new_counting(
    limits.max_canonical_bytes,
  )
  encode_model_plan_canonical(
    counter,
    shape_constraints,
    kv_cache_geometry,
    workspace_bounds,
    operations,
    required,
    final_output,
    kv_execution,
    numeric_schema.digest(),
  )
  limits.check_canonical_bytes(counter.count)
  counter.count
}

///|
fn derive_model_identity(
  content_digest : @spec.ContentDigest,
  expected_canonical_bytes : Int,
  shape_constraints : ShapeConstraints,
  kv_cache_geometry : KvCacheGeometry,
  workspace_bounds : WorkspaceBounds,
  operations : ArrayView[PlanOperation],
  required_capabilities : ArrayView[KernelCapabilityId],
  final_output : ActivationRef,
  kv_execution : ModelKvExecution,
  numeric_schema_digest : @numeric_contract.ModelNumericSchemaDigest,
) -> @spec.ModelIdentity raise PlanValidationError {
  let writer = ModelPlanCanonicalWriter::new_hashing(expected_canonical_bytes)
  encode_model_plan_canonical(
    writer, shape_constraints, kv_cache_geometry, workspace_bounds, operations, required_capabilities,
    final_output, kv_execution, numeric_schema_digest,
  )
  if writer.count != expected_canonical_bytes {
    raise CanonicalEncodingMismatch
  }
  let plan_digest = try! @spec.PlanDigest::from_sha256(writer.finish_hex())
  try! @spec.ModelIdentity::new(content_digest, plan_digest)
}
