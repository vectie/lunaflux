///|
fn validate_projection_operands(
  operation : @plan.PlanOperation,
  parameters : ProjectionParameters,
  profile : @launch_contract.PagedKernelProfile,
  operands : ArrayView[@launch_contract.AotOperand],
  dispatch : @luna_projection_strategy.ProjectionDispatchPlan,
) -> Unit raise ProjectionAotError {
  let roles = projection_roles(
    operation,
    projection_decode_workspace_bytes(parameters, dispatch) > 0L,
  )
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
      RuntimeInput(QueryRowOffsets) =>
        projection_checked_mul((profile.max_query_rows() + 1).to_int64(), 4L)
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
      Workspace => projection_workspace_bytes(parameters, profile, dispatch)
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
  dispatch : @luna_projection_strategy.ProjectionDispatchPlan,
  sibling_ownership~ : @luna_projection_tile_compiler.ProjectionSiblingOwnership = SplitSiblingPlanes,
) -> @launch_contract.AotLaunchDimensions raise ProjectionAotError {
  let tensor_core = projection_tensor_core_eligible(parameters, dispatch)
  let tensor_core_mlp = projection_tensor_core_mlp_eligible(
    parameters, dispatch,
  )
  let groups = dispatch.prefill().output_tiles_per_workgroup().to_int64()
  let gather = @luna_projection_tile_compiler.projection_selected_row_gather(
    dispatch,
    if parameters.family == LanguageModelHead {
      QueryRowEnds
    } else {
      AllTokenRows
    },
  )
  let shared_memory_bytes = if tensor_core_mlp &&
    sibling_ownership != SplitSiblingPlanes {
    // Register-retained and coowned epilogues do not use the dynamic peer
    // plane. Their checked static operand ring owns any required peer storage.
    0L
  } else if tensor_core && gather is Some(_) {
    0L
  } else if tensor_core || tensor_core_mlp {
    let storage = dispatch
      .prefill()
      .local_storage(projection_strategy_shape(parameters)) catch {
        _ => raise SizeOverflow(LaunchDimensions)
      }
    let accumulator_bytes = projection_checked_mul(
      storage.accumulator_elements(),
      4L,
    )
    let input_elements = storage.selected_input_elements()
    let input_bytes = if input_elements == 0L {
      0L
    } else {
      projection_checked_mul(input_elements, 2L)
    }
    if accumulator_bytes > 2147483647L - input_bytes {
      raise LimitExceeded(LaunchDimensions)
    }
    accumulator_bytes + input_bytes
  } else if parameters.family == GatedMlp &&
    parameters.intermediate_width <= 24576 {
    projection_checked_mul(parameters.intermediate_width.to_int64(), 4L)
  } else {
    0L
  }
  let work = if tensor_core {
    if parameters.family == LanguageModelHead {
      let output_tile_groups = (
          parameters.output_width.to_int64() + groups * 16L - 1L
        ) /
        (groups * 16L)
      output_tile_groups
    } else {
      let output_tile_groups = (
          parameters.output_width.to_int64() + groups * 16L - 1L
        ) /
        (groups * 16L)
      let row_tiles = (profile.max_query_tokens().to_int64() + 15L) / 16L
      let prompt_work = projection_checked_mul(output_tile_groups, row_tiles)
      let decode_work = (parameters.output_width.to_int64() + groups - 1L) /
        groups
      if prompt_work > decode_work {
        prompt_work
      } else {
        decode_work
      }
    }
  } else if tensor_core_mlp {
    let row_tiles = (profile.max_query_tokens().to_int64() + 15L) / 16L
    let intermediate_groups = parameters.intermediate_width.to_int64() /
      (groups * 16L)
    Int64::max(
      projection_checked_mul(row_tiles, intermediate_groups),
      (
        Int32::max(parameters.intermediate_width, parameters.output_width).to_int64() +
        groups -
        1L
      ) /
      groups,
    )
  } else if shared_memory_bytes > 0L {
    profile.max_query_tokens().to_int64()
  } else {
    projection_checked_mul(
      (if parameters.family == LanguageModelHead {
        profile.max_query_rows()
      } else {
        profile.max_query_tokens()
      }).to_int64(),
      parameters.output_width.to_int64(),
    )
  }
  let threads = if tensor_core || tensor_core_mlp { groups * 32L } else { 256L }
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
