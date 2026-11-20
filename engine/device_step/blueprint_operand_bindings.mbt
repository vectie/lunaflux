///|
fn runtime_stage(input : @plan.OperationRuntimeInput) -> DeviceStepStage {
  match input {
    StepCounts => Counts
    QueryPositions => TokenPositions
    QueryRowOffsets => QueryRowOffsets
    SequenceLengths => SequenceLengths
    PageTableOffsets => PageOffsets
    PageTablePageIndices => PhysicalPageIndices
  }
}

///|
fn runtime_operand_bytes(
  input : @plan.OperationRuntimeInput,
  profile : @launch_contract.PagedKernelProfile,
) -> Int64 {
  match input {
    StepCounts => 20L
    QueryPositions => profile.max_query_tokens().to_int64() * 4L
    QueryRowOffsets | PageTableOffsets =>
      (profile.max_query_rows().to_int64() + 1L) * 4L
    SequenceLengths => profile.max_query_rows().to_int64() * 4L
    PageTablePageIndices => profile.max_page_table_entries().to_int64() * 4L
  }
}

///|
fn activation_is_bound(
  binding : @device_memory.OperationMemoryBinding,
  reference : @plan.ActivationRef,
  output : Bool,
) -> Bool {
  let regions = if output {
    binding.activation_outputs()
  } else {
    binding.activation_inputs()
  }
  regions.any(region => region.reference() == reference)
}

///|
fn bind_activation_operand(
  operation : @plan.OperationId,
  reference : @plan.ActivationRef,
  output : Bool,
  operand : @launch_contract.AotOperand,
  profile : @launch_contract.PagedKernelProfile,
  memory_plan : @device_memory.DeviceMemoryPlan,
) -> PagedKvOperandBinding raise DeviceStepError {
  guard memory_plan.operation_binding(operation) is Some(operation_binding) &&
    activation_is_bound(operation_binding, reference, output) &&
    memory_plan.region(reference) is Some(region) else {
    raise blueprint_operation_error(operation, ActivationRegion)
  }
  let envelope_rows = memory_plan.token_rows()
  let capacity = region.logical_bytes()
  let live_rows = profile.max_query_tokens().to_int64()
  if capacity <= 0L ||
    capacity % envelope_rows != 0L ||
    live_rows > envelope_rows ||
    operand.byte_count() != capacity / envelope_rows * live_rows ||
    operand.byte_count() > region.reserved_bytes() {
    raise blueprint_operation_error(operation, OperandSize)
  }
  let physical_alignment = memory_plan.limits().activation_alignment()
  if operand.alignment() > physical_alignment ||
    physical_alignment % operand.alignment() != 0L ||
    region.offset_bytes() % operand.alignment() != 0L ||
    region.end_offset_bytes() > memory_plan.activation_bytes() {
    raise blueprint_operation_error(operation, OperandAlignment)
  }
  {
    role: operand.role(),
    space: ActivationWorkspaceArena,
    offset_bytes: region.offset_bytes(),
    byte_count: operand.byte_count(),
    alignment: operand.alignment(),
  }
}

///|
fn bind_workspace_operand(
  operation : @plan.OperationId,
  operand : @launch_contract.AotOperand,
  memory_plan : @device_memory.DeviceMemoryPlan,
) -> PagedKvOperandBinding raise DeviceStepError {
  let workspace = memory_plan.workspace()
  if operand.byte_count() > workspace.byte_length() ||
    workspace.end_offset_bytes() > memory_plan.arena_bytes() {
    raise blueprint_operation_error(operation, OperandSize)
  }
  if operand.alignment() > workspace.alignment_bytes() ||
    workspace.alignment_bytes() % operand.alignment() != 0L ||
    workspace.offset_bytes() % operand.alignment() != 0L {
    raise blueprint_operation_error(operation, OperandAlignment)
  }
  {
    role: operand.role(),
    space: ActivationWorkspaceArena,
    offset_bytes: workspace.offset_bytes(),
    byte_count: operand.byte_count(),
    alignment: operand.alignment(),
  }
}

///|
fn bind_kv_operand(
  operation : @plan.OperationId,
  layer : @plan.KvLayerId,
  component : @device_layout.DeviceKvComponent,
  operand : @launch_contract.AotOperand,
  layout : @device_layout.DeviceKvLayout,
) -> PagedKvOperandBinding raise DeviceStepError {
  let region = layout.region(layer.as_int(), component, 0) catch {
    _ => raise blueprint_operation_error(operation, PersistentKvRegion)
  }
  if operand.byte_count() != layout.component_stride_bytes() ||
    region.offset_bytes() > layout.arena_bytes() - operand.byte_count() {
    raise blueprint_operation_error(operation, OperandSize)
  }
  if operand.alignment() != layout.segment_alignment_bytes() ||
    region.offset_bytes() % operand.alignment() != 0L {
    raise blueprint_operation_error(operation, OperandAlignment)
  }
  {
    role: operand.role(),
    space: PersistentKvArena,
    offset_bytes: region.offset_bytes(),
    byte_count: operand.byte_count(),
    alignment: operand.alignment(),
  }
}

///|
fn bind_blueprint_operand(
  operation : @plan.OperationId,
  operand : @launch_contract.AotOperand,
  profile : @launch_contract.PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
  memory_plan : @device_memory.DeviceMemoryPlan,
  static_operation : @device_plan.StaticOperation?,
  static_plan : @device_plan.StaticDevicePlan?,
) -> PagedKvOperandBinding raise DeviceStepError {
  match operand.role() {
    RuntimeInput(input) => {
      if operand.byte_count() != runtime_operand_bytes(input, profile) {
        raise blueprint_operation_error(operation, OperandSize)
      }
      if operand.alignment() != 4L {
        raise blueprint_operation_error(operation, OperandAlignment)
      }
      {
        role: operand.role(),
        space: StepDescriptor(runtime_stage(input)),
        offset_bytes: 0L,
        byte_count: operand.byte_count(),
        alignment: operand.alignment(),
      }
    }
    ActivationInput(reference) =>
      bind_activation_operand(
        operation, reference, false, operand, profile, memory_plan,
      )
    ActivationOutput(reference) =>
      bind_activation_operand(
        operation, reference, true, operand, profile, memory_plan,
      )
    PersistentKvComponent(layer, component) =>
      bind_kv_operand(operation, layer, component, operand, layout)
    Workspace => bind_workspace_operand(operation, operand, memory_plan)
    ExternalInput(reference) =>
      bind_external_operand(
        operation, reference, operand, profile, static_operation,
      )
    WeightInput(reference) =>
      bind_weight_operand(
        operation, reference, operand, static_operation, static_plan,
      )
    WeightScaleInput(_, _) =>
      raise blueprint_operation_error(operation, OperandRole)
  }
}
