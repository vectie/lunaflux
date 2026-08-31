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
/// CUDA's private capability adapter into the backend-neutral strategy layer.
fn projection_dispatch_plan(
  parameters : ProjectionParameters,
  maximum_rows : Int,
) -> @luna_projection_strategy.ProjectionDispatchPlan {
  let family = match parameters.family {
    QkvProjection => @luna_projection_strategy.Qkv
    DenseProjection => Dense
    GatedMlp => GatedMlp
    LanguageModelHead => LanguageModelHead
  }
  let shape = @luna_projection_strategy.ProjectionShape::new(
    family,
    input_width=parameters.input_width,
    output_width=parameters.output_width,
    intermediate_width=parameters.intermediate_width,
  ) catch {
    _ => abort("validated projection shape rejected by strategy")
  }
  let matrix_tile = if parameters.family == GatedMlp {
    if parameters.input_width % 256 == 0 &&
      parameters.output_width % 256 == 0 &&
      parameters.intermediate_width % 256 == 0 &&
      parameters.intermediate_width <= 3840 {
      Some(
        @luna_projection_strategy.ProjectionTile::new(
          rows=16,
          columns=256,
          reduction=256,
        ) catch {
          _ => abort("static CUDA MLP tile rejected")
        },
      )
    } else {
      None
    }
  } else if parameters.input_width % 16 == 0 &&
    parameters.output_width % 16 == 0 &&
    (
      parameters.family != QkvProjection ||
      (
        parameters.query_width > 0 &&
        parameters.query_width % 16 == 0 &&
        parameters.intermediate_width > 0 &&
        parameters.intermediate_width % 16 == 0
      )
    ) {
    Some(
      @luna_projection_strategy.ProjectionTile::new(
        rows=16,
        columns=16,
        reduction=16,
      ) catch {
        _ => abort("static CUDA matrix tile rejected")
      },
    )
  } else {
    None
  }
  let capabilities = @luna_projection_strategy.ProjectionCapabilities::new(
    subgroup_width=32,
    matrix_tile~,
  ) catch {
    _ => abort("static CUDA projection capability rejected")
  }
  @luna_projection_strategy.plan_projection_dispatch(
    shape, maximum_rows, capabilities,
  ) catch {
    _ => abort("validated projection strategy rejected")
  }
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
) -> Array[@launch_contract.AotOperandRole] raise ProjectionAotError {
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
  if projection_decode_workspace_bytes(projection_parameters(operation)) > 0L {
    roles.push(Workspace)
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
      Workspace => projection_workspace_bytes(parameters, profile)
      _ => raise Mismatch(OperandOrder)
    }
    if operand.byte_count() != expected_bytes {
      raise Mismatch(OperandSize)
    }
    let expected_alignment = match role {
      RuntimeInput(_) => 4L
      Workspace => PROJECTION_VALUE_ALIGNMENT
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
  parameters : ProjectionParameters,
) -> @launch_contract.AotLaunchDimensions raise ProjectionAotError {
  let tensor_core = projection_tensor_core_eligible(parameters)
  let tensor_core_mlp = projection_tensor_core_mlp_eligible(parameters)
  let shared_memory_bytes = if tensor_core {
    8192L
  } else if tensor_core_mlp {
    16384L
  } else if parameters.family == GatedMlp &&
    parameters.intermediate_width <= 24576 {
    projection_checked_mul(parameters.intermediate_width.to_int64(), 4L)
  } else {
    0L
  }
  let work = if tensor_core {
    let output_tile_groups = (parameters.output_width.to_int64() + 127L) / 128L
    if parameters.family == LanguageModelHead {
      output_tile_groups
    } else {
      let row_tiles = (profile.max_query_tokens().to_int64() + 15L) / 16L
      let prompt_work = projection_checked_mul(output_tile_groups, row_tiles)
      let decode_work = (parameters.output_width.to_int64() + 7L) / 8L
      if prompt_work > decode_work {
        prompt_work
      } else {
        decode_work
      }
    }
  } else if tensor_core_mlp {
    let row_tiles = (profile.max_query_tokens().to_int64() + 15L) / 16L
    let intermediate_groups = parameters.intermediate_width.to_int64() / 256L
    projection_checked_mul(row_tiles, intermediate_groups)
  } else if shared_memory_bytes > 0L {
    profile.max_query_tokens().to_int64()
  } else {
    projection_checked_mul(
      profile.max_query_tokens().to_int64(),
      parameters.output_width.to_int64(),
    )
  }
  let threads = if tensor_core_mlp { 512L } else { 256L }
  if work > 9223372036854775807L - (threads - 1L) {
    raise SizeOverflow(LaunchDimensions)
  }
  let grid = if tensor_core || tensor_core_mlp || shared_memory_bytes > 0L {
    work
  } else {
    (work + threads - 1L) / threads
  }
  if grid <= 0L || grid > 2147483647L {
    raise LimitExceeded(LaunchDimensions)
  }
  @launch_contract.AotLaunchDimensions::new(
    grid_x=grid.to_int(),
    grid_y=1,
    grid_z=1,
    block_x=threads.to_int(),
    block_y=1,
    block_z=1,
    shared_memory_bytes=shared_memory_bytes.to_int(),
  ) catch {
    _ => raise Invalid(LaunchDimensions)
  }
}

///|
fn validate_projection_contract(
  operation : @plan.PlanOperation,
  contract : @launch_contract.PagedKvAotLaunchContract,
) -> Unit raise ProjectionAotError {
  let parameters = projection_bind_qkv_widths(
    operation,
    projection_parameters(operation),
    contract.operands(),
  )
  let workspace_bytes = projection_workspace_bytes(
    parameters,
    contract.profile(),
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
  let expected = projection_dimensions(contract.profile(), parameters)
  if contract.dimensions() != expected {
    raise Mismatch(LaunchDimensions)
  }
}
