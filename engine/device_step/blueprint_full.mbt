///|
fn static_input_matches(
  operation : @device_plan.StaticOperation,
  expected : @device_plan.StaticInput,
) -> Bool {
  operation.inputs().any(input => input == expected)
}

///|
fn bind_external_operand(
  operation : @plan.OperationId,
  reference : @plan.ModelInputRef,
  operand : @launch_contract.AotOperand,
  profile : @launch_contract.PagedKernelProfile,
  static_operation : @device_plan.StaticOperation?,
) -> PagedKvOperandBinding raise DeviceStepError {
  guard reference == TokenIds &&
    static_operation is Some(owner) &&
    static_input_matches(owner, ExternalInput(reference)) else {
    raise blueprint_operation_error(operation, OperandRole)
  }
  if operand.byte_count() != profile.max_query_tokens().to_int64() * 4L {
    raise blueprint_operation_error(operation, OperandSize)
  }
  if operand.alignment() != 4L {
    raise blueprint_operation_error(operation, OperandAlignment)
  }
  {
    role: operand.role(),
    space: StepDescriptor(TokenIds),
    offset_bytes: 0L,
    byte_count: operand.byte_count(),
    alignment: operand.alignment(),
  }
}

///|
fn bind_weight_operand(
  operation : @plan.OperationId,
  reference : @plan.TensorRef,
  operand : @launch_contract.AotOperand,
  static_operation : @device_plan.StaticOperation?,
  static_plan : @device_plan.StaticDevicePlan?,
) -> PagedKvOperandBinding raise DeviceStepError {
  guard static_operation is Some(owner) && static_plan is Some(plan) else {
    raise blueprint_operation_error(operation, OperandRole)
  }
  let mut matched : @device_materialize.DeviceWeightRegion? = None
  for input in owner.inputs() {
    if input is WeightRegion(region) && region.reference() == reference {
      matched = Some(region)
    }
  }
  guard matched is Some(region) else {
    raise blueprint_operation_error(operation, OperandRole)
  }
  if operand.byte_count() != region.byte_length() ||
    region.end_offset_bytes() > plan.weight_arena_bytes() {
    raise blueprint_operation_error(operation, OperandSize)
  }
  if operand.alignment() > plan.weight_alignment() ||
    plan.weight_alignment() % operand.alignment() != 0L ||
    region.offset_bytes() % operand.alignment() != 0L {
    raise blueprint_operation_error(operation, OperandAlignment)
  }
  {
    role: operand.role(),
    space: WeightArena,
    offset_bytes: region.offset_bytes(),
    byte_count: operand.byte_count(),
    alignment: operand.alignment(),
  }
}

///|
fn validate_full_static_plan(
  model_plan : @plan.ModelPlan,
  static_plan : @device_plan.StaticDevicePlan,
  contracts : @launch_contract.PagedKvLaunchContractSet,
) -> Unit raise DeviceStepError {
  if static_plan.identity() != model_plan.identity() {
    raise blueprint_error(Identity)
  }
  if static_plan.target() != contracts.device_target() {
    raise blueprint_error(Target)
  }
  if static_plan.catalog_version() != contracts.catalog_version() ||
    static_plan.weight_arena_bytes() <= 0L ||
    static_plan.weight_alignment() <= 0L ||
    static_plan.weight_tensor_count() <= 0 {
    raise blueprint_error(CatalogVersion)
  }
  if static_plan.final_output() != model_plan.final_output() {
    raise blueprint_error(OperationOrder)
  }
  let semantic = model_plan.operations()
  let physical = static_plan.operations()
  if semantic.length() != physical.length() {
    raise blueprint_error(OperationCount)
  }
  for index, operation in semantic {
    let bound = physical[index]
    if bound.id() != operation.id() ||
      bound.kind() != operation.kind() ||
      bound.shape() != operation.shape() ||
      bound.outputs() != operation.outputs() ||
      bound.binding().operation_id() != operation.id() ||
      bound.binding().capability() != operation.required_capability() ||
      bound.binding().semantic_version() != @catalog.KernelSemanticVersion::v3() {
      raise blueprint_operation_error(operation.id(), ContractSemantics)
    }
  }
}

///|
/// Admits an inert physical blueprint for every operation in one paged graph.
///
/// The exact static plan supplies already-validated weight regions while the
/// memory plan supplies activation/workspace placement. The result contains no
/// allocation, module, function, kernel argument, device address, or lifecycle
/// authority.
pub fn admit_paged_graph_execution_blueprint(
  model_plan : @plan.ModelPlan,
  limits : DeviceStepLimits,
  layout : @device_layout.DeviceKvLayout,
  static_plan : @device_plan.StaticDevicePlan,
  memory_plan : @device_memory.DeviceMemoryPlan,
  profile_id : @launch_contract.KernelProfileId,
  contracts : @launch_contract.PagedKvLaunchContractSet,
  artifacts : @artifact.KernelArtifactBundle,
) -> PagedKvExecutionBlueprint raise DeviceStepError {
  validate_blueprint_evidence(
    model_plan,
    limits,
    layout,
    memory_plan,
    contracts,
    artifacts,
    FullGraph,
  )
  validate_full_static_plan(model_plan, static_plan, contracts)
  let profile = selected_paged_profile(contracts, profile_id, limits)
  let operations = model_plan.operations()
  let selected = selected_paged_contracts(contracts, profile)
  if selected.length() != operations.length() {
    raise blueprint_error(OperationCount)
  }
  let steps : Array[PagedKvExecutionStep] = []
  for index, operation in operations {
    if selected[index].operation_id() != operation.id() {
      raise blueprint_operation_error(operation.id(), OperationOrder)
    }
    steps.push(
      build_paged_step(
        operation,
        selected[index],
        profile,
        layout,
        memory_plan,
        artifacts,
        Some(static_plan.operations()[index]),
        Some(static_plan),
      ),
    )
  }
  {
    identity: model_plan.identity(),
    target: contracts.device_target(),
    catalog_version: contracts.catalog_version(),
    scope: contracts.scope(),
    profile,
    limits,
    layout,
    weight_arena_bytes: static_plan.weight_arena_bytes(),
    weight_alignment: static_plan.weight_alignment(),
    activation_workspace_arena_bytes: memory_plan.arena_bytes(),
    steps: ReadOnlyArray::from_array(steps),
  }
}
