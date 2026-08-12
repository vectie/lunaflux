///|
const MAX_OPERATIONS : Int = 4096

///|
const MAX_INPUTS_PER_OPERATION : Int = 64

///|
const MAX_TOTAL_INPUTS : Int = 8192

///|
const MAX_WEIGHT_REGIONS : Int = 65536

///|
const MAX_INT64 : Int64 = 9223372036854775807L

///|
fn is_power_of_two(value : Int64) -> Bool {
  value > 0L && (value & (value - 1L)) == 0L
}

///|
fn checked_workspace_minimum(
  byte_count : Int64,
  alignment : Int64,
  operation_id : @plan.OperationId,
) -> Int64 raise DevicePlanError {
  let remainder = byte_count % alignment
  if remainder == 0L {
    byte_count
  } else {
    let padding = alignment - remainder
    if byte_count > MAX_INT64 - padding {
      raise WorkspaceOverflow(operation_id)
    }
    byte_count + padding
  }
}

///|
fn validate_workspace(
  operation : @plan.PlanOperation,
  binding : @catalog.KernelBinding,
  model_workspace_bound : Int64,
) -> Unit raise DevicePlanError {
  let operation_id = operation.id()
  let alignment = binding.workspace_alignment()
  let byte_count = binding.workspace_bytes()
  if !is_power_of_two(alignment) || alignment > 65536L {
    raise WorkspaceMismatch(operation_id, InvalidAlignment)
  }
  if byte_count < 0L {
    raise WorkspaceMismatch(operation_id, NegativeSize)
  }
  if byte_count % alignment != 0L {
    raise WorkspaceMismatch(operation_id, MisalignedSize)
  }
  let required = checked_workspace_minimum(
    operation.workspace_bytes(),
    alignment,
    operation_id,
  )
  if byte_count < required {
    raise WorkspaceMismatch(operation_id, InsufficientForOperation)
  }
  if byte_count > model_workspace_bound {
    raise WorkspaceMismatch(operation_id, ExceedsModelBound)
  }
}

///|
fn weight_region_error(
  tensor : @plan.TensorRef,
  region : @device_materialize.DeviceWeightRegion,
  arena_bytes : Int64,
  arena_alignment : Int64,
) -> WeightRegionRule? {
  if region.reference() != tensor {
    return Some(ReferenceMismatch)
  }
  let offset = region.offset_bytes()
  let byte_length = region.byte_length()
  if offset < 0L {
    return Some(NegativeOffset)
  }
  if byte_length <= 0L {
    return Some(NonPositiveLength)
  }
  if offset % arena_alignment != 0L {
    return Some(MisalignedOffset)
  }
  if offset > MAX_INT64 - byte_length {
    return Some(EndOverflow)
  }
  if offset + byte_length > arena_bytes {
    return Some(OutsideArena)
  }
  None
}

///|
fn validate_weight_layout(
  layout : @device_materialize.DeviceWeightLayout,
) -> Unit raise DevicePlanError {
  if layout.arena_bytes() <= 0L {
    raise InvalidWeightArena(NonPositiveSize)
  }
  let alignment = layout.alignment_bytes()
  if alignment < 16L || alignment > 4096L || !is_power_of_two(alignment) {
    raise InvalidWeightArena(InvalidAlignment)
  }
  if layout.length() <= 0 {
    raise InvalidWeightArena(EmptyRegionSet)
  }
  if layout.length() > MAX_WEIGHT_REGIONS {
    raise InvalidWeightArena(RegionCount)
  }
  let mut previous_end = 0L
  for index in 0..<layout.length() {
    let tensor = try! @plan.TensorRef::new(index)
    guard layout.region(tensor) is Some(region) else {
      raise MissingLayoutRegion(tensor)
    }
    match weight_region_error(tensor, region, layout.arena_bytes(), alignment) {
      Some(rule) => raise InvalidLayoutRegion(tensor, rule)
      None => ()
    }
    if region.offset_bytes() < previous_end {
      raise InvalidLayoutRegion(tensor, OverlappingPrevious)
    }
    previous_end = region.end_offset_bytes()
  }
}

///|
fn validate_weight_region(
  operation_id : @plan.OperationId,
  tensor : @plan.TensorRef,
  region : @device_materialize.DeviceWeightRegion,
  arena_bytes : Int64,
  arena_alignment : Int64,
) -> Unit raise DevicePlanError {
  match weight_region_error(tensor, region, arena_bytes, arena_alignment) {
    Some(rule) => raise InvalidWeightRegion(operation_id, tensor, rule)
    None => ()
  }
}

///|
fn resolve_inputs(
  operation : @plan.PlanOperation,
  layout : @device_materialize.DeviceWeightLayout,
) -> ReadOnlyArray[StaticInput] raise DevicePlanError {
  let resolved : Array[StaticInput] = []
  for input in operation.inputs() {
    match input {
      ModelInput(reference) => resolved.push(ExternalInput(reference))
      Activation(reference) => resolved.push(ActivationValue(reference))
      Tensor(reference) => {
        guard layout.region(reference) is Some(region) else {
          raise MissingWeightRegion(operation.id(), reference)
        }
        validate_weight_region(
          operation.id(),
          reference,
          region,
          layout.arena_bytes(),
          layout.alignment_bytes(),
        )
        resolved.push(WeightRegion(region))
      }
    }
  }
  ReadOnlyArray::from_array(resolved)
}

///|
/// Builds a bounded immutable device-execution plan from three independently
/// validated startup inputs.
///
/// This function performs no model-family dispatch, backend selection, device
/// allocation, kernel loading, or execution. Every kernel choice must already
/// be exact in `resolved_catalog`; all incompatibilities fail here at startup.
pub fn build(
  model_plan : @plan.ModelPlan,
  weight_layout : @device_materialize.DeviceWeightLayout,
  resolved_catalog : @catalog.ResolvedKernelCatalog,
) -> StaticDevicePlan raise DevicePlanError {
  if weight_layout.identity() != model_plan.identity() {
    raise IdentityMismatch(WeightLayout)
  }
  if resolved_catalog.model_identity() != model_plan.identity() {
    raise IdentityMismatch(KernelCatalog)
  }
  validate_weight_layout(weight_layout)
  let operations = model_plan.operations()
  let bindings = resolved_catalog.bindings()
  if operations.length() > MAX_OPERATIONS {
    raise BoundExceeded(OperationCount)
  }
  if bindings.length() != operations.length() {
    raise BindingCountMismatch
  }
  let mut total_inputs = 0
  let mut max_workspace_bytes = 0L
  let mut max_workspace_alignment = 1L
  let static_operations : Array[StaticOperation] = []
  for index, operation in operations {
    let input_count = operation.inputs().length()
    if input_count > MAX_INPUTS_PER_OPERATION {
      raise BoundExceeded(InputsPerOperation)
    }
    if total_inputs > MAX_TOTAL_INPUTS - input_count {
      raise BoundExceeded(TotalInputCount)
    }
    total_inputs = total_inputs + input_count
    let binding = bindings[index]
    if binding.operation_id() != operation.id() {
      raise BindingOrderMismatch(operation.id())
    }
    if binding.capability() != operation.required_capability() {
      raise BindingCapabilityMismatch(operation.id())
    }
    validate_workspace(
      operation,
      binding,
      model_plan.workspace_bounds().max_temporary_bytes(),
    )
    if binding.workspace_bytes() > max_workspace_bytes {
      max_workspace_bytes = binding.workspace_bytes()
    }
    if binding.workspace_alignment() > max_workspace_alignment {
      max_workspace_alignment = binding.workspace_alignment()
    }
    static_operations.push({
      id: operation.id(),
      kind: operation.kind(),
      shape: operation.shape(),
      inputs: resolve_inputs(operation, weight_layout),
      outputs: operation.outputs(),
      binding,
    })
  }
  if resolved_catalog.max_workspace_bytes() != max_workspace_bytes {
    raise WorkspaceSummaryMismatch(SummarySize)
  }
  if resolved_catalog.max_workspace_alignment() != max_workspace_alignment {
    raise WorkspaceSummaryMismatch(SummaryAlignment)
  }
  {
    identity: model_plan.identity(),
    shape_constraints: model_plan.shape_constraints(),
    target: resolved_catalog.device_target(),
    catalog_version: resolved_catalog.catalog_version(),
    weight_arena_bytes: weight_layout.arena_bytes(),
    weight_alignment: weight_layout.alignment_bytes(),
    weight_tensor_count: weight_layout.length(),
    workspace_bytes: max_workspace_bytes,
    workspace_alignment: max_workspace_alignment,
    operations: ReadOnlyArray::from_array(static_operations),
    final_output: model_plan.final_output(),
  }
}
