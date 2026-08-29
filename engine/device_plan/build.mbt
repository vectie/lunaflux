///|
const MAX_OPERATIONS : Int = 4096

///|
const MAX_INPUTS_PER_OPERATION : Int = 64

///|
const MAX_TOTAL_INPUTS : Int = 16384

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
fn validate_graph_bounds(
  operations : ReadOnlyArray[@plan.PlanOperation],
) -> Unit raise DevicePlanError {
  if operations.length() > MAX_OPERATIONS {
    raise BoundExceeded(OperationCount)
  }
  let mut total_inputs = 0
  for operation in operations {
    let input_count = operation.inputs().length()
    if input_count > MAX_INPUTS_PER_OPERATION {
      raise BoundExceeded(InputsPerOperation)
    }
    if total_inputs > MAX_TOTAL_INPUTS - input_count {
      raise BoundExceeded(TotalInputCount)
    }
    total_inputs += input_count
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
  validate_plan_identities(model_plan, weight_layout, resolved_catalog)
  let operations = model_plan.operations()
  let numeric_binding = model_plan.numeric_binding()
  let numeric_operations = numeric_binding.operations()
  let bindings = resolved_catalog.bindings()
  validate_graph_bounds(operations)
  if bindings.length() != operations.length() {
    raise BindingCountMismatch
  }
  if numeric_operations.length() != operations.length() {
    raise NumericBindingCountMismatch
  }
  validate_weight_layout(model_plan, weight_layout)
  let mut max_workspace_bytes = 0L
  let mut max_workspace_alignment = 1L
  let static_operations : Array[StaticOperation] = []
  for index, operation in operations {
    let binding = bindings[index]
    if binding.operation_id() != operation.id() {
      raise BindingOrderMismatch(operation.id())
    }
    if binding.capability() != operation.required_capability() {
      raise BindingCapabilityMismatch(operation.id())
    }
    let numeric_operation = numeric_operations[index]
    if numeric_operation.operation() != operation.id() {
      raise NumericBindingOrderMismatch(operation.id())
    }
    let execution_digest = numeric_operation.execution().digest()
    validate_kernel_numeric_binding(
      operation.id(),
      execution_digest,
      resolved_catalog.catalog_version(),
      binding,
    )
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
      operation_execution_digest: execution_digest,
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
    numeric_schema_digest: numeric_binding.digest(),
    weight_layout,
    workspace_bytes: max_workspace_bytes,
    workspace_alignment: max_workspace_alignment,
    operations: ReadOnlyArray::from_array(static_operations),
    final_output: model_plan.final_output(),
  }
}
