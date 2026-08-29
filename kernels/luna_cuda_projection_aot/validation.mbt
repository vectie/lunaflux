///|
const PROJECTION_BF16_BYTES : Int64 = 2L

///|
const PROJECTION_STEP_COUNTS_BYTES : Int64 = 20L

///|
const PROJECTION_VALUE_ALIGNMENT : Int64 = 16L

///|
priv struct ProjectionParameters {
  family : ProjectionAotFamily
  input_width : Int
  query_width : Int
  output_width : Int
  intermediate_width : Int
}

///|
fn projection_checked_mul(
  left : Int64,
  right : Int64,
) -> Int64 raise ProjectionAotError {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    raise SizeOverflow(Shape)
  }
  left * right
}

///|
fn projection_matrix_bytes(
  rows : Int,
  columns : Int,
) -> Int64 raise ProjectionAotError {
  projection_checked_mul(
    projection_checked_mul(rows.to_int64(), columns.to_int64()),
    PROJECTION_BF16_BYTES,
  )
}

///|
fn projection_exact_runtime_inputs(
  actual : ReadOnlyArray[@plan.OperationRuntimeInput],
) -> Bool {
  actual.length() == 1 && actual[0] == StepCounts
}

///|
fn projection_is_activation(value : @plan.ValueRef) -> Bool {
  value is Activation(_)
}

///|
fn projection_is_weight(value : @plan.ValueRef) -> Bool {
  value is Tensor(_)
}

///|
fn projection_common_operation(
  operation : @plan.PlanOperation,
) -> Unit raise ProjectionAotError {
  if operation.outputs().length() != 1 ||
    operation.workspace_bytes() != 0L ||
    operation.kv_layer() is Some(_) ||
    operation.persistent_state() != NoPersistentState ||
    !projection_exact_runtime_inputs(operation.runtime_inputs()) {
    raise Mismatch(Operation)
  }
}

///|
fn projection_parameters(
  operation : @plan.PlanOperation,
) -> ProjectionParameters raise ProjectionAotError {
  projection_common_operation(operation)
  match (operation.kind(), operation.shape()) {
    (QkvProjection, ProjectionShape(input_width~, output_width~)) => {
      if operation.required_capability() !=
        @plan.KernelCapabilityId::qkv_projection() ||
        operation.inputs().length() != 4 ||
        !projection_is_activation(operation.inputs()[0]) ||
        !projection_is_weight(operation.inputs()[1]) ||
        !projection_is_weight(operation.inputs()[2]) ||
        !projection_is_weight(operation.inputs()[3]) ||
        input_width <= 0 ||
        output_width <= 0 {
        raise Mismatch(Operation)
      }
      {
        family: QkvProjection,
        input_width,
        query_width: 0,
        output_width,
        intermediate_width: 0,
      }
    }
    (
      OutputProjection
      | LanguageModelHead,
      ProjectionShape(input_width~, output_width~),
    ) => {
      if operation.required_capability() !=
        @plan.KernelCapabilityId::dense_projection() ||
        operation.inputs().length() != 2 ||
        !projection_is_activation(operation.inputs()[0]) ||
        !projection_is_weight(operation.inputs()[1]) ||
        input_width <= 0 ||
        output_width <= 0 {
        raise Mismatch(Operation)
      }
      {
        family: if operation.kind() == OutputProjection {
          DenseProjection
        } else {
          LanguageModelHead
        },
        input_width,
        query_width: 0,
        output_width,
        intermediate_width: 0,
      }
    }
    (GatedMlp, GatedMlpShape(hidden_size~, intermediate_size~)) => {
      if operation.required_capability() !=
        @plan.KernelCapabilityId::gated_mlp() ||
        operation.inputs().length() != 4 ||
        !projection_is_activation(operation.inputs()[0]) ||
        !projection_is_weight(operation.inputs()[1]) ||
        !projection_is_weight(operation.inputs()[2]) ||
        !projection_is_weight(operation.inputs()[3]) ||
        hidden_size <= 0 ||
        intermediate_size <= 0 {
        raise Mismatch(Operation)
      }
      {
        family: GatedMlp,
        input_width: hidden_size,
        query_width: 0,
        output_width: hidden_size,
        intermediate_width: intermediate_size,
      }
    }
    _ => raise Unsupported(Operation)
  }
}

///|
fn projection_qkv_width_from_bytes(
  byte_count : Int64,
  input_width : Int,
) -> Int raise ProjectionAotError {
  let row_bytes = projection_checked_mul(
    input_width.to_int64(),
    PROJECTION_BF16_BYTES,
  )
  if byte_count <= 0L || byte_count % row_bytes != 0L {
    raise Mismatch(OperandSize)
  }
  let rows = byte_count / row_bytes
  if rows <= 0L || rows > 2147483647L {
    raise SizeOverflow(Shape)
  }
  rows.to_int()
}

///|
fn projection_bind_qkv_widths(
  operation : @plan.PlanOperation,
  parameters : ProjectionParameters,
  operands : ArrayView[@launch_contract.AotOperand],
) -> ProjectionParameters raise ProjectionAotError {
  if parameters.family != QkvProjection {
    return parameters
  }
  let roles = projection_roles(operation)
  if operands.length() != roles.length() || operands.length() != 6 {
    raise Mismatch(OperandOrder)
  }
  for index, operand in operands {
    if operand.role() != roles[index] {
      raise Mismatch(OperandOrder)
    }
  }
  let query_width = projection_qkv_width_from_bytes(
    operands[2].byte_count(),
    parameters.input_width,
  )
  let key_width = projection_qkv_width_from_bytes(
    operands[3].byte_count(),
    parameters.input_width,
  )
  let value_width = projection_qkv_width_from_bytes(
    operands[4].byte_count(),
    parameters.input_width,
  )
  if key_width != value_width ||
    query_width.to_int64() + 2L * key_width.to_int64() !=
    parameters.output_width.to_int64() {
    raise Mismatch(OperandSize)
  }
  { ..parameters, query_width, intermediate_width: key_width }
}

///|
fn projection_roles(
  operation : @plan.PlanOperation,
) -> Array[@launch_contract.AotOperandRole] {
  let roles : Array[@launch_contract.AotOperandRole] = [
    RuntimeInput(StepCounts),
  ]
  for input in operation.inputs() {
    roles.push(
      match input {
        Activation(reference) => ActivationInput(reference)
        Tensor(reference) => WeightInput(reference)
        ModelInput(reference) => ExternalInput(reference)
      },
    )
  }
  for output in operation.outputs() {
    roles.push(ActivationOutput(output))
  }
  roles
}

///|
fn projection_weight_bytes(
  family : ProjectionAotFamily,
  parameters : ProjectionParameters,
  weight_index : Int,
) -> Int64 raise ProjectionAotError {
  match family {
    QkvProjection =>
      match weight_index {
        0 =>
          projection_matrix_bytes(
            parameters.query_width,
            parameters.input_width,
          )
        1 | 2 =>
          projection_matrix_bytes(
            parameters.intermediate_width,
            parameters.input_width,
          )
        _ => raise Mismatch(OperandOrder)
      }
    DenseProjection | LanguageModelHead =>
      if weight_index == 0 {
        projection_matrix_bytes(parameters.output_width, parameters.input_width)
      } else {
        raise Mismatch(OperandOrder)
      }
    GatedMlp =>
      match weight_index {
        0 | 1 =>
          projection_matrix_bytes(
            parameters.intermediate_width,
            parameters.input_width,
          )
        2 =>
          projection_matrix_bytes(
            parameters.output_width,
            parameters.intermediate_width,
          )
        _ => raise Mismatch(OperandOrder)
      }
  }
}

///|
fn validate_projection_operands(
  operation : @plan.PlanOperation,
  parameters : ProjectionParameters,
  profile : @launch_contract.PagedKernelProfile,
  operands : ArrayView[@launch_contract.AotOperand],
) -> Unit raise ProjectionAotError {
  let roles = projection_roles(operation)
  if operands.length() != roles.length() {
    raise Mismatch(OperandOrder)
  }
  let input_bytes = projection_matrix_bytes(
    profile.max_query_tokens(),
    parameters.input_width,
  )
  let output_bytes = projection_matrix_bytes(
    profile.max_query_tokens(),
    parameters.output_width,
  )
  let mut weight_index = 0
  for index, operand in operands {
    let role = roles[index]
    if operand.role() != role {
      raise Mismatch(OperandOrder)
    }
    let expected_bytes = match role {
      RuntimeInput(StepCounts) => PROJECTION_STEP_COUNTS_BYTES
      ActivationInput(_) => input_bytes
      WeightInput(_) => {
        let result = projection_weight_bytes(
          parameters.family,
          parameters,
          weight_index,
        )
        weight_index += 1
        result
      }
      ActivationOutput(_) => output_bytes
      _ => raise Mismatch(OperandOrder)
    }
    if operand.byte_count() != expected_bytes {
      raise Mismatch(OperandSize)
    }
    let expected_alignment = match role {
      RuntimeInput(_) => 4L
      _ => PROJECTION_VALUE_ALIGNMENT
    }
    if operand.alignment() != expected_alignment {
      raise Mismatch(OperandAlignment)
    }
  }
}

///|
fn projection_dimensions(
  profile : @launch_contract.PagedKernelProfile,
  output_width : Int,
) -> @launch_contract.AotLaunchDimensions raise ProjectionAotError {
  let work = projection_checked_mul(
    profile.max_query_tokens().to_int64(),
    output_width.to_int64(),
  )
  let threads = 256L
  if work > 9223372036854775807L - (threads - 1L) {
    raise SizeOverflow(LaunchDimensions)
  }
  let grid = (work + threads - 1L) / threads
  if grid <= 0L || grid > 2147483647L {
    raise LimitExceeded(LaunchDimensions)
  }
  @launch_contract.AotLaunchDimensions::new(
    grid_x=grid.to_int(),
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    _ => raise Invalid(LaunchDimensions)
  }
}

///|
fn validate_projection_contract(
  operation : @plan.PlanOperation,
  contract : @launch_contract.PagedKvAotLaunchContract,
) -> Unit raise ProjectionAotError {
  if contract.operation_id() != operation.id() ||
    contract.operation_kind() != operation.kind() ||
    contract.operation_shape() != operation.shape() ||
    contract.capability() != operation.required_capability() ||
    contract.semantic_version() != @catalog.KernelSemanticVersion::v3() ||
    contract.kv_layer() != operation.kv_layer() ||
    contract.runtime_inputs() != operation.runtime_inputs() ||
    contract.persistent_state() != operation.persistent_state() ||
    contract.workspace_bytes() != operation.workspace_bytes() ||
    contract.workspace_bytes() != 0L ||
    contract.workspace_alignment() <= 0L ||
    (contract.workspace_alignment() & (contract.workspace_alignment() - 1L)) !=
    0L {
    raise Mismatch(Operation)
  }
  let expected = projection_dimensions(
    contract.profile(),
    projection_bind_qkv_widths(
      operation,
      projection_parameters(operation),
      contract.operands(),
    ).output_width,
  )
  if contract.dimensions() != expected {
    raise Mismatch(LaunchDimensions)
  }
}
