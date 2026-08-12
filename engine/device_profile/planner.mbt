///|
fn region_id_for_weight(
  reference : @plan.TensorRef,
) -> ProfileRegionId raise DeviceProfileError {
  checked_region_id_add(1L, Int64::from_int(reference.as_int()))
}

///|
fn activation_region_base(static_plan : @device_plan.StaticDevicePlan) -> Int64 {
  1L + Int64::from_int(static_plan.weight_tensor_count())
}

///|
fn region_id_for_activation(
  static_plan : @device_plan.StaticDevicePlan,
  reference : @plan.ActivationRef,
) -> ProfileRegionId raise DeviceProfileError {
  checked_region_id_add(
    activation_region_base(static_plan),
    Int64::from_int(reference.as_int()),
  )
}

///|
fn region_id_for_workspace(
  static_plan : @device_plan.StaticDevicePlan,
  activation_count : Int,
) -> ProfileRegionId raise DeviceProfileError {
  checked_region_id_add(
    activation_region_base(static_plan),
    Int64::from_int(activation_count),
  )
}

///|
fn append_region_once(
  regions : Array[ExecutionRegionView],
  view : ExecutionRegionView,
) -> Unit {
  for existing in regions {
    if existing.id == view.id {
      return
    }
  }
  regions.push(view)
}

///|
fn validate_shape_envelope(
  static_plan : @device_plan.StaticDevicePlan,
  shape : ExactExecutionShape,
) -> Unit raise DeviceProfileError {
  let constraints = static_plan.shape_constraints()
  if shape.batch_rows() > constraints.max_batch_rows() {
    raise InvalidShape(BatchExceedsEnvelope)
  }
  if shape.query_tokens_per_row() > constraints.max_sequence_tokens() {
    raise InvalidShape(QueryExceedsEnvelope)
  }
}

///|
fn maximum_token_rows(
  static_plan : @device_plan.StaticDevicePlan,
) -> Int64 raise DeviceProfileError {
  let operations = static_plan.operations()
  let operation = operations[0].id()
  checked_multiply(
    Int64::from_int(static_plan.shape_constraints().max_batch_rows()),
    Int64::from_int(static_plan.shape_constraints().max_sequence_tokens()),
    operation,
  )
}

///|
fn validate_plan_pair(
  static_plan : @device_plan.StaticDevicePlan,
  memory_plan : @device_memory.DeviceMemoryPlan,
) -> Unit raise DeviceProfileError {
  if static_plan.identity() != memory_plan.identity() {
    raise InvalidPlan(Identity)
  }
  if static_plan.target() != memory_plan.target() {
    raise InvalidPlan(Target)
  }
  if static_plan.operations().length() == 0 {
    raise InvalidPlan(EmptyOperations)
  }
  if static_plan.operations().length() != memory_plan.operation_count() {
    raise InvalidPlan(OperationCount)
  }
  if static_plan.operations().length() != memory_plan.activation_count() {
    raise InvalidPlan(ActivationCount)
  }
  if memory_plan.token_rows() != maximum_token_rows(static_plan) {
    raise InvalidPlan(MaximumTokenRows)
  }
  if static_plan.final_output() != memory_plan.final_output() {
    raise InvalidPlan(FinalOutput)
  }
  let workspace = memory_plan.workspace()
  let workspace_view = {
    id: region_id_for_workspace(static_plan, memory_plan.activation_count()),
    space: ActivationWorkspaceArena,
    offset_bytes: workspace.offset_bytes(),
    logical_bytes: workspace.byte_length(),
    reserved_bytes: workspace.byte_length(),
    alignment_bytes: workspace.alignment_bytes(),
  }
  validate_region_capacity(workspace_view, memory_plan.arena_bytes())
  if workspace.end_offset_bytes() != memory_plan.arena_bytes() {
    raise InvalidPlan(Workspace)
  }
  if memory_plan.activation_bytes() > workspace.offset_bytes() ||
    memory_plan.arena_bytes() > memory_plan.limits().max_arena_bytes() {
    raise InvalidPlan(Workspace)
  }
}

///|
fn weight_view(
  static_plan : @device_plan.StaticDevicePlan,
  region : @device_materialize.DeviceWeightRegion,
) -> ExecutionRegionView raise DeviceProfileError {
  let reference_index = region.reference().as_int()
  if reference_index < 0 || reference_index >= static_plan.weight_tensor_count() {
    raise InvalidRegion(region_id_for_weight(region.reference()), Missing)
  }
  let view = {
    id: region_id_for_weight(region.reference()),
    space: WeightArena,
    offset_bytes: region.offset_bytes(),
    logical_bytes: region.byte_length(),
    reserved_bytes: region.byte_length(),
    alignment_bytes: static_plan.weight_alignment(),
  }
  validate_region_capacity(view, static_plan.weight_arena_bytes())
  if view.logical_bytes == 0L {
    raise InvalidRegion(view.id, EmptyRequiredRegion)
  }
  view
}

///|
fn activation_view(
  static_plan : @device_plan.StaticDevicePlan,
  memory_plan : @device_memory.DeviceMemoryPlan,
  operation : @device_plan.StaticOperation,
  width : Int64,
  exact_token_rows : Int64,
  maximum_rows : Int64,
) -> ActivationValueView raise DeviceProfileError {
  let reference = operation.outputs()[0]
  guard memory_plan.region(reference) is Some(memory_region) else {
    raise InvalidRegion(
      region_id_for_activation(static_plan, reference),
      Missing,
    )
  }
  if memory_region.reference() != reference ||
    memory_region.produced_by() != operation.id() {
    raise InvalidOperation(operation.id(), MemoryBinding)
  }
  let exact_bytes = checked_multiply(
    checked_multiply(width, exact_token_rows, operation.id()),
    BF16_BYTES,
    operation.id(),
  )
  let maximum_bytes = checked_multiply(
    checked_multiply(width, maximum_rows, operation.id()),
    BF16_BYTES,
    operation.id(),
  )
  if memory_region.logical_bytes() != maximum_bytes {
    raise InvalidOperation(operation.id(), MemoryBinding)
  }
  let view = {
    id: region_id_for_activation(static_plan, reference),
    space: ActivationWorkspaceArena,
    offset_bytes: memory_region.offset_bytes(),
    logical_bytes: exact_bytes,
    reserved_bytes: memory_region.reserved_bytes(),
    alignment_bytes: memory_plan.limits().activation_alignment(),
  }
  validate_region_capacity(view, memory_plan.activation_bytes())
  { reference, width, token_rows: exact_token_rows, region: view }
}

///|
fn workspace_view_for_operation(
  static_plan : @device_plan.StaticDevicePlan,
  memory_plan : @device_memory.DeviceMemoryPlan,
  operation : @device_plan.StaticOperation,
) -> ExecutionRegionView raise DeviceProfileError {
  let requested_bytes = operation.binding().workspace_bytes()
  let requested_alignment = operation.binding().workspace_alignment()
  let workspace = memory_plan.workspace()
  if requested_bytes < 0L ||
    requested_bytes > workspace.byte_length() ||
    !is_power_of_two(requested_alignment) ||
    workspace.offset_bytes() % requested_alignment != 0L {
    raise InvalidOperation(operation.id(), Workspace)
  }
  let view = {
    id: region_id_for_workspace(static_plan, memory_plan.activation_count()),
    space: ActivationWorkspaceArena,
    offset_bytes: workspace.offset_bytes(),
    logical_bytes: requested_bytes,
    reserved_bytes: workspace.byte_length(),
    alignment_bytes: requested_alignment,
  }
  validate_region_capacity(view, memory_plan.arena_bytes())
  view
}

///|
fn validate_memory_binding(
  operation : @device_plan.StaticOperation,
  memory_binding : @device_memory.OperationMemoryBinding,
  memory_plan : @device_memory.DeviceMemoryPlan,
) -> Unit raise DeviceProfileError {
  if memory_binding.operation_id() != operation.id() {
    raise InvalidOperation(operation.id(), MemoryBinding)
  }
  let mut activation_input_index = 0
  for input in operation.inputs() {
    if input is ActivationValue(reference) {
      guard memory_binding.activation_inputs().get(activation_input_index)
        is Some(region) &&
        region.reference() == reference else {
        raise InvalidOperation(operation.id(), ActivationInputOrder)
      }
      if memory_plan.region(reference) != Some(region) {
        raise InvalidOperation(operation.id(), MemoryBinding)
      }
      activation_input_index += 1
    }
  }
  if activation_input_index != memory_binding.activation_inputs().length() ||
    memory_binding.activation_outputs().length() != 1 ||
    memory_binding.activation_outputs()[0].reference() != operation.outputs()[0] ||
    memory_plan.region(operation.outputs()[0]) !=
    Some(memory_binding.activation_outputs()[0]) ||
    memory_binding.workspace() != memory_plan.workspace() {
    raise InvalidOperation(operation.id(), MemoryBinding)
  }
}

///|
fn build_inputs(
  static_plan : @device_plan.StaticDevicePlan,
  operation : @device_plan.StaticOperation,
  token_staging : ExecutionRegionView,
  activations : Array[ActivationValueView],
  regions : Array[ExecutionRegionView],
) -> ReadOnlyArray[OperationInputView] raise DeviceProfileError {
  if !valid_input_roles(operation) {
    raise InvalidOperation(operation.id(), InputRoles)
  }
  let expected_widths = expected_activation_widths(operation)
  let inputs : Array[OperationInputView] = []
  let mut activation_index = 0
  for input in operation.inputs() {
    match input {
      ExternalInput(TokenIds) => inputs.push(TokenIds(token_staging))
      WeightRegion(weight_region) => {
        let view = weight_view(static_plan, weight_region)
        append_region_once(regions, view)
        inputs.push(Weight(weight_region.reference(), view))
      }
      ActivationValue(reference) => {
        guard activations.get(reference.as_int()) is Some(value) else {
          raise InvalidOperation(operation.id(), ActivationInput)
        }
        guard expected_widths.get(activation_index) is Some(expected_width) else {
          raise InvalidOperation(operation.id(), InputRoles)
        }
        if value.width() != expected_width {
          raise InvalidOperation(operation.id(), ActivationWidth)
        }
        inputs.push(Activation(value))
        activation_index += 1
      }
    }
  }
  if activation_index != expected_widths.length() {
    raise InvalidOperation(operation.id(), InputRoles)
  }
  ReadOnlyArray::from_array(inputs)
}

///|
fn build_final_output(
  static_plan : @device_plan.StaticDevicePlan,
  shape : ExactExecutionShape,
  activations : Array[ActivationValueView],
  memory_plan : @device_memory.DeviceMemoryPlan,
) -> FinalOutputView raise DeviceProfileError {
  let reference = static_plan.final_output()
  guard activations.get(reference.as_int()) is Some(value) else {
    raise InvalidPlan(FinalOutput)
  }
  let producer_index = value.reference().as_int()
  guard static_plan.operations().get(producer_index) is Some(producer) else {
    raise InvalidPlan(FinalOutput)
  }
  let row_bytes = checked_multiply(value.width(), BF16_BYTES, producer.id())
  let rows : Array[FinalRowView] = []
  for batch in 0..<shape.batch_rows() {
    let flat_row = checked_add(
      checked_multiply(
        Int64::from_int(batch),
        Int64::from_int(shape.query_tokens_per_row()),
        producer.id(),
      ),
      Int64::from_int(shape.query_tokens_per_row() - 1),
      producer.id(),
    )
    let row_offset = checked_add(
      value.region().offset_bytes(),
      checked_multiply(flat_row, row_bytes, producer.id()),
      producer.id(),
    )
    let row_region = {
      id: value.region().id(),
      space: ActivationWorkspaceArena,
      offset_bytes: row_offset,
      logical_bytes: row_bytes,
      reserved_bytes: row_bytes,
      alignment_bytes: BF16_BYTES,
    }
    validate_region_capacity(row_region, memory_plan.activation_bytes())
    let row_end = checked_add(row_offset, row_bytes, producer.id())
    let value_end = checked_add(
      value.region().offset_bytes(),
      value.region().logical_bytes(),
      producer.id(),
    )
    if row_end > value_end {
      raise InvalidRegion(row_region.id, OutsideArena)
    }
    rows.push({
      batch_index: batch,
      backing_region: value.region().id(),
      offset_bytes: row_offset,
      byte_length: row_bytes,
      alignment_bytes: BF16_BYTES,
    })
  }
  {
    semantics: if producer.kind() == LanguageModelHead {
      LanguageModelLogits
    } else {
      ActivationOutput
    },
    value,
    row_bytes,
    selected_rows: ReadOnlyArray::from_array(rows),
  }
}

///|
/// Builds a bounded exact-shape profile over preplanned reusable storage.
///
/// This is startup/preparation work. A prepared executor must resolve kernel
/// code and vendor plans against `shape.kernel_shape()` exactly; equality of
/// tensor widths or fit within the maximum arena is not evidence that another
/// request shape is executable by the same compiled kernel.
pub fn build(
  static_plan : @device_plan.StaticDevicePlan,
  memory_plan : @device_memory.DeviceMemoryPlan,
  shape : ExactExecutionShape,
) -> DeviceExecutionProfile raise DeviceProfileError {
  validate_plan_pair(static_plan, memory_plan)
  validate_shape_envelope(static_plan, shape)
  let maximum_rows = memory_plan.token_rows()
  let token_capacity = checked_token_bytes(maximum_rows)
  let token_staging = {
    id: ProfileRegionId(0),
    space: TokenInput,
    offset_bytes: 0L,
    logical_bytes: checked_token_bytes(shape.token_rows()),
    reserved_bytes: token_capacity,
    alignment_bytes: TOKEN_ID_BYTES,
  }
  validate_region_capacity(token_staging, token_capacity)
  let regions : Array[ExecutionRegionView] = [token_staging]
  let activations : Array[ActivationValueView] = []
  let operation_profiles : Array[OperationExecutionProfile] = []
  for index, operation in static_plan.operations() {
    if operation.id().as_int() != index {
      raise InvalidPlan(OperationOrder)
    }
    if operation.outputs().length() != 1 {
      raise InvalidOperation(operation.id(), OutputArity)
    }
    if operation.outputs()[0].as_int() != index {
      raise InvalidOperation(operation.id(), OutputOrder)
    }
    guard memory_plan.operation_binding(operation.id()) is Some(memory_binding) else {
      raise InvalidOperation(operation.id(), MemoryBinding)
    }
    validate_memory_binding(operation, memory_binding, memory_plan)
    let inputs = build_inputs(
      static_plan, operation, token_staging, activations, regions,
    )
    let output = activation_view(
      static_plan,
      memory_plan,
      operation,
      operation_output_width(operation),
      shape.token_rows(),
      maximum_rows,
    )
    activations.push(output)
    append_region_once(regions, output.region())
    let workspace = workspace_view_for_operation(
      static_plan, memory_plan, operation,
    )
    operation_profiles.push({
      operation_id: operation.id(),
      kernel_shape: shape.kernel_shape(),
      inputs,
      outputs: ReadOnlyArray::from_array([output]),
      workspace,
    })
  }
  let workspace = memory_plan.workspace()
  append_region_once(regions, {
    id: region_id_for_workspace(static_plan, memory_plan.activation_count()),
    space: ActivationWorkspaceArena,
    offset_bytes: workspace.offset_bytes(),
    logical_bytes: workspace.byte_length(),
    reserved_bytes: workspace.byte_length(),
    alignment_bytes: workspace.alignment_bytes(),
  })
  {
    identity: static_plan.identity(),
    target: static_plan.target(),
    shape,
    token_staging,
    token_staging_capacity_bytes: token_capacity,
    regions: ReadOnlyArray::from_array(regions),
    operations: ReadOnlyArray::from_array(operation_profiles),
    final_output: build_final_output(
      static_plan, shape, activations, memory_plan,
    ),
  }
}
