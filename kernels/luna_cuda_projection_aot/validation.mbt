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
  rows : Int32,
  columns : Int32,
) -> Int64 raise ProjectionAotError {
  projection_checked_mul(
    projection_checked_mul(rows.to_int64(), columns.to_int64()),
    PROJECTION_BF16_BYTES,
  )
}

///|
fn projection_exact_runtime_inputs(
  family : ProjectionAotFamily,
  actual : ReadOnlyArray[@plan.OperationRuntimeInput],
) -> Bool {
  if family == LanguageModelHead {
    actual.length() == 2 &&
    actual[0] == StepCounts &&
    actual[1] == QueryRowOffsets
  } else {
    actual.length() == 1 && actual[0] == StepCounts
  }
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
    operation.persistent_state() != NoPersistentState {
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
        !projection_exact_runtime_inputs(
          QkvProjection,
          operation.runtime_inputs(),
        ) ||
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
        !projection_exact_runtime_inputs(
          if operation.kind() == LanguageModelHead {
            LanguageModelHead
          } else {
            DenseProjection
          },
          operation.runtime_inputs(),
        ) ||
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
        !projection_exact_runtime_inputs(GatedMlp, operation.runtime_inputs()) ||
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
  input_width : Int32,
) -> Int32 raise ProjectionAotError {
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
  let roles = projection_roles(operation, false)
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
  guard @luna_projection_tile_compiler.plan_projection_concatenated_rows([
      query_width, key_width, value_width,
    ])
    is Some(segments) else {
    raise Mismatch(OperandSize)
  }
  if key_width != value_width || segments[2].row_end != parameters.output_width {
    raise Mismatch(OperandSize)
  }
  { ..parameters, query_width, intermediate_width: key_width, }
}

///|
fn projection_roles(
  operation : @plan.PlanOperation,
  include_workspace : Bool,
) -> Array[@launch_contract.AotOperandRole] {
  let roles : Array[@launch_contract.AotOperandRole] = []
  for input in operation.runtime_inputs() {
    roles.push(RuntimeInput(input))
  }
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
  if include_workspace {
    roles.push(Workspace)
  }
  roles
}

///|
fn projection_weight_bytes(
  family : ProjectionAotFamily,
  parameters : ProjectionParameters,
  weight_index : Int32,
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
fn validate_projection_contract(
  operation : @plan.PlanOperation,
  contract : @launch_contract.PagedKvAotLaunchContract,
  dispatch : @luna_projection_strategy.ProjectionDispatchPlan,
  compiled_dimensions : @launch_contract.AotLaunchDimensions,
) -> Unit raise ProjectionAotError {
  let parameters = projection_bind_qkv_widths(
    operation,
    projection_parameters(operation),
    contract.operands(),
  )
  let workspace_bytes = projection_workspace_bytes(
    parameters,
    contract.profile(),
    dispatch,
  )
  if contract.operation_id() != operation.id() ||
    contract.operation_kind() != operation.kind() ||
    contract.operation_shape() != operation.shape() ||
    contract.capability() != operation.required_capability() ||
    contract.semantic_version() != @catalog.KernelSemanticVersion::v3() ||
    contract.kv_layer() != operation.kv_layer() ||
    contract.runtime_inputs() != operation.runtime_inputs() ||
    contract.persistent_state() != operation.persistent_state() ||
    contract.workspace_bytes() < operation.workspace_bytes() ||
    contract.workspace_bytes() != workspace_bytes ||
    contract.workspace_alignment() <= 0L ||
    (contract.workspace_alignment() & (contract.workspace_alignment() - 1L)) !=
    0L {
    raise Mismatch(Operation)
  }
  // Launch geometry belongs to the immutable scheduled lowering, not to the
  // untuned semantic dispatch. Rebuilding the latter here rejects legal AOT
  // row/output products selected by the fold compiler. The caller owns the
  // private lowering; external contracts still must match it exactly.
  if contract.dimensions() != compiled_dimensions {
    raise Mismatch(LaunchDimensions)
  }
}
