///|
const BF16_BYTES : Int64 = 2L

///|
const INT32_BYTES : Int64 = 4L

///|
const STEP_COUNT_BYTES : Int64 = 20L

///|
priv struct PointwiseParameters {
  family : PointwiseAotFamily
  width : Int
  vocabulary_size : Int
  query_heads : Int
  kv_heads : Int
  head_dimension : Int
  epsilon : Double
  rope_theta : Double
  tokens_per_page : Int
  total_page_count : Int
  page_stride_bytes : Int64
}

///|
fn checked_mul(left : Int64, right : Int64) -> Int64 raise PointwiseAotError {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    raise SizeOverflow(Shape)
  }
  left * right
}

///|
fn bf16_matrix_bytes(
  rows : Int,
  columns : Int,
) -> Int64 raise PointwiseAotError {
  checked_mul(checked_mul(rows.to_int64(), columns.to_int64()), BF16_BYTES)
}

///|
fn qkv_width(
  query_heads : Int,
  kv_heads : Int,
  head_dimension : Int,
) -> Int raise PointwiseAotError {
  if query_heads <= 0 ||
    kv_heads <= 0 ||
    head_dimension <= 0 ||
    kv_heads > query_heads ||
    query_heads % kv_heads != 0 ||
    head_dimension % 2 != 0 {
    raise Unsupported(Shape)
  }
  let heads = query_heads.to_int64() + 2L * kv_heads.to_int64()
  let width = checked_mul(heads, head_dimension.to_int64())
  if width > 2147483647L {
    raise SizeOverflow(Shape)
  }
  width.to_int()
}

///|
fn exact_runtime_inputs(
  actual : ReadOnlyArray[@plan.OperationRuntimeInput],
  expected : ArrayView[@plan.OperationRuntimeInput],
) -> Bool {
  if actual.length() != expected.length() {
    return false
  }
  for index, value in actual {
    if value != expected[index] {
      return false
    }
  }
  true
}

///|
fn common_pointwise_operation(
  operation : @plan.PlanOperation,
  expected_capability : @plan.KernelCapabilityId,
) -> Unit raise PointwiseAotError {
  if operation.required_capability() != expected_capability ||
    operation.workspace_bytes() != 0L ||
    operation.outputs().length() != 1 ||
    operation.persistent_state() != NoPersistentState {
    raise Mismatch(Operation)
  }
}

///|
fn pointwise_parameters(
  operation : @plan.PlanOperation,
) -> PointwiseParameters raise PointwiseAotError {
  match (operation.kind(), operation.shape()) {
    (Embedding, EmbeddingShape(vocabulary_size~, width~)) => {
      common_pointwise_operation(
        operation,
        @plan.KernelCapabilityId::embedding_lookup(),
      )
      if vocabulary_size <= 0 || width <= 0 {
        raise Unsupported(Shape)
      }
      if !exact_runtime_inputs(operation.runtime_inputs(), [StepCounts]) ||
        operation.inputs().length() != 2 ||
        operation.inputs()[0] != ModelInput(TokenIds) ||
        !(operation.inputs()[1] is Tensor(_)) {
        raise Mismatch(Operation)
      }
      {
        family: EmbeddingLookup,
        width,
        vocabulary_size,
        query_heads: 0,
        kv_heads: 0,
        head_dimension: 0,
        epsilon: 0.0,
        rope_theta: 0.0,
        tokens_per_page: 0,
        total_page_count: 0,
        page_stride_bytes: 0L,
      }
    }
    (RmsNorm, NormalizationShape(width~, epsilon~)) => {
      common_pointwise_operation(
        operation,
        @plan.KernelCapabilityId::rms_norm(),
      )
      if width <= 0 {
        raise Unsupported(Shape)
      }
      if !exact_runtime_inputs(operation.runtime_inputs(), [StepCounts]) ||
        operation.inputs().length() != 2 ||
        !(operation.inputs()[0] is Activation(_)) ||
        !(operation.inputs()[1] is Tensor(_)) ||
        epsilon.is_nan() ||
        epsilon.is_inf() ||
        epsilon <= 0.0 {
        raise Mismatch(Operation)
      }
      {
        family: RmsNorm,
        width,
        vocabulary_size: 0,
        query_heads: 0,
        kv_heads: 0,
        head_dimension: 0,
        epsilon,
        rope_theta: 0.0,
        tokens_per_page: 0,
        total_page_count: 0,
        page_stride_bytes: 0L,
      }
    }
    (
      QkRmsNorm,
      QkNormalizationShape(query_heads~, kv_heads~, head_dimension~, epsilon~),
    ) => {
      common_pointwise_operation(
        operation,
        @plan.KernelCapabilityId::qk_rms_norm(),
      )
      let width = qkv_width(query_heads, kv_heads, head_dimension)
      if !exact_runtime_inputs(operation.runtime_inputs(), [StepCounts]) ||
        operation.inputs().length() != 3 ||
        !(operation.inputs()[0] is Activation(_)) ||
        !(operation.inputs()[1] is Tensor(_)) ||
        !(operation.inputs()[2] is Tensor(_)) ||
        epsilon.is_nan() ||
        epsilon.is_inf() ||
        epsilon <= 0.0 ||
        epsilon > 1.0 {
        raise Mismatch(Operation)
      }
      {
        family: QkRmsNorm,
        width,
        vocabulary_size: 0,
        query_heads,
        kv_heads,
        head_dimension,
        epsilon,
        rope_theta: 0.0,
        tokens_per_page: 0,
        total_page_count: 0,
        page_stride_bytes: 0L,
      }
    }
    (ResidualAdd, ElementwiseShape(width~)) => {
      common_pointwise_operation(
        operation,
        @plan.KernelCapabilityId::residual_add(),
      )
      if width <= 0 {
        raise Unsupported(Shape)
      }
      if !exact_runtime_inputs(operation.runtime_inputs(), [StepCounts]) ||
        operation.inputs().length() != 2 ||
        !(operation.inputs()[0] is Activation(_)) ||
        !(operation.inputs()[1] is Activation(_)) {
        raise Mismatch(Operation)
      }
      {
        family: ResidualAdd,
        width,
        vocabulary_size: 0,
        query_heads: 0,
        kv_heads: 0,
        head_dimension: 0,
        epsilon: 0.0,
        rope_theta: 0.0,
        tokens_per_page: 0,
        total_page_count: 0,
        page_stride_bytes: 0L,
      }
    }
    (
      RotaryEmbedding,
      RotaryShape(query_heads~, kv_heads~, head_dimension~, rope_theta~),
    ) => {
      common_pointwise_operation(
        operation,
        @plan.KernelCapabilityId::positioned_rotary_embedding(),
      )
      if !exact_runtime_inputs(operation.runtime_inputs(), [
          StepCounts,
          QueryPositions,
        ]) ||
        operation.inputs().length() != 1 ||
        !(operation.inputs()[0] is Activation(_)) ||
        operation.kv_layer() is None ||
        rope_theta.is_nan() ||
        rope_theta.is_inf() ||
        rope_theta <= 0.0 {
        raise Mismatch(Operation)
      }
      let width = qkv_width(query_heads, kv_heads, head_dimension)
      if query_heads + kv_heads > 32 {
        raise Unsupported(Shape)
      }
      {
        family: PositionedRotary,
        width,
        vocabulary_size: 0,
        query_heads,
        kv_heads,
        head_dimension,
        epsilon: 0.0,
        rope_theta,
        tokens_per_page: 0,
        total_page_count: 0,
        page_stride_bytes: 0L,
      }
    }
    _ => raise Unsupported(Operation)
  }
}

///|
fn canonical_manifest_roles(
  operation : @plan.PlanOperation,
) -> Array[@launch_contract.AotOperandRole] {
  let roles : Array[@launch_contract.AotOperandRole] = []
  for runtime in operation.runtime_inputs() {
    roles.push(RuntimeInput(runtime))
  }
  for input in operation.inputs() {
    roles.push(
      match input {
        ModelInput(reference) => ExternalInput(reference)
        Tensor(reference) => WeightInput(reference)
        Activation(reference) => ActivationInput(reference)
      },
    )
  }
  for output in operation.outputs() {
    roles.push(ActivationOutput(output))
  }
  roles
}

///|
fn runtime_bytes(
  role : @plan.OperationRuntimeInput,
  profile : @launch_contract.PagedKernelProfile,
) -> Int64 {
  match role {
    StepCounts => STEP_COUNT_BYTES
    QueryPositions => profile.max_query_tokens().to_int64() * INT32_BYTES
    QueryRowOffsets | PageTableOffsets =>
      (profile.max_query_rows().to_int64() + 1L) * INT32_BYTES
    SequenceLengths => profile.max_query_rows().to_int64() * INT32_BYTES
    PageTablePageIndices =>
      profile.max_page_table_entries().to_int64() * INT32_BYTES
  }
}

///|
fn manifest_operand_bytes(
  role : @launch_contract.AotOperandRole,
  parameters : PointwiseParameters,
  profile : @launch_contract.PagedKernelProfile,
) -> Int64 raise PointwiseAotError {
  match role {
    RuntimeInput(runtime) => runtime_bytes(runtime, profile)
    ExternalInput(TokenIds) =>
      profile.max_query_tokens().to_int64() * INT32_BYTES
    WeightInput(_) =>
      match parameters.family {
        EmbeddingLookup =>
          bf16_matrix_bytes(parameters.vocabulary_size, parameters.width)
        RmsNorm => bf16_matrix_bytes(1, parameters.width)
        QkRmsNorm => bf16_matrix_bytes(1, parameters.head_dimension)
        _ => raise Mismatch(OperandOrder)
      }
    ActivationInput(_) | ActivationOutput(_) =>
      bf16_matrix_bytes(profile.max_query_tokens(), parameters.width)
    _ => raise Mismatch(OperandOrder)
  }
}

///|
fn validate_operands(
  operation : @plan.PlanOperation,
  parameters : PointwiseParameters,
  profile : @launch_contract.PagedKernelProfile,
  operands : ArrayView[@launch_contract.AotOperand],
) -> Unit raise PointwiseAotError {
  let roles = canonical_manifest_roles(operation)
  if operands.length() != roles.length() {
    raise Mismatch(OperandOrder)
  }
  for index, operand in operands {
    let role = roles[index]
    if operand.role() != role {
      raise Mismatch(OperandOrder)
    }
    if operand.byte_count() != manifest_operand_bytes(role, parameters, profile) {
      raise Mismatch(OperandSize)
    }
    let minimum_alignment = match role {
      RuntimeInput(_) | ExternalInput(_) => INT32_BYTES
      _ => BF16_BYTES
    }
    if operand.alignment() < minimum_alignment ||
      operand.alignment() % minimum_alignment != 0L {
      raise Mismatch(OperandAlignment)
    }
  }
}
