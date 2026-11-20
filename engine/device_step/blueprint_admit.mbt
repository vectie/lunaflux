///|
fn build_paged_step(
  operation : @plan.PlanOperation,
  contract : @launch_contract.PagedKvAotLaunchContract,
  profile : @launch_contract.PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
  memory_plan : @device_memory.DeviceMemoryPlan,
  artifacts : @artifact.KernelArtifactBundle,
  static_operation : @device_plan.StaticOperation?,
  static_plan : @device_plan.StaticDevicePlan?,
) -> PagedKvExecutionStep raise DeviceStepError {
  let layer = operation.kv_layer()
  guard memory_plan.operation_binding(operation.id()) is Some(memory_binding) &&
    memory_binding.operation_id() == operation.id() else {
    raise blueprint_operation_error(operation.id(), ActivationRegion)
  }
  if memory_binding.workspace() != memory_plan.workspace() {
    raise blueprint_operation_error(operation.id(), WorkspaceRegion)
  }
  if contract.operation_id() != operation.id() ||
    contract.profile() != profile ||
    contract.operation_kind() != operation.kind() ||
    contract.operation_shape() != operation.shape() ||
    contract.capability() != operation.required_capability() ||
    contract.semantic_version() != @catalog.KernelSemanticVersion::v3() ||
    contract.kv_layer() != layer ||
    contract.runtime_inputs() != operation.runtime_inputs() ||
    contract.persistent_state() != operation.persistent_state() ||
    contract.kv_layout_version() != layout.stable_layout_version() ||
    contract.tokens_per_page() != layout.tokens_per_page() ||
    contract.workspace_bytes() < operation.workspace_bytes() {
    raise blueprint_operation_error(operation.id(), ContractSemantics)
  }
  let expected_roles = expected_paged_roles(
    operation,
    contract.workspace_bytes(),
  )
  let source_operands = contract.operands()
  if source_operands.length() != expected_roles.length() {
    raise blueprint_operation_error(operation.id(), OperandCount)
  }
  let operands : Array[PagedKvOperandBinding] = []
  for index, operand in source_operands {
    if operand.role() != expected_roles[index] {
      raise blueprint_operation_error(operation.id(), OperandRole)
    }
    operands.push(
      bind_blueprint_operand(
        operation.id(),
        operand,
        profile,
        layout,
        memory_plan,
        static_operation,
        static_plan,
      ),
    )
  }
  guard artifacts.entry_point(contract.entry_point()) is Some(entry) else {
    raise blueprint_operation_error(operation.id(), ArtifactEntryPoint)
  }
  let module_index = entry.module_index()
  let workspace = source_operands.any(operand => operand.role() == Workspace)
  validate_projection_companion(
    operation.id(),
    operation.kind(),
    workspace,
    entry.companion_dimensions(),
  )
  if !entry.row_variants().is_empty() &&
    operation.kind() != QkvProjection &&
    operation.kind() != OutputProjection &&
    operation.kind() != LanguageModelHead &&
    operation.kind() != GatedMlp {
    raise blueprint_operation_error(operation.id(), ArtifactEntryPoint)
  }
  for variant in entry.row_variants() {
    validate_projection_companion(
      operation.id(),
      operation.kind(),
      workspace,
      variant.companion_dimensions(),
    )
    let dimensions = variant.dimensions()
    if variant.bounded_work_grid_x() is Some(grid) &&
      (grid <= 0 || grid > dimensions.grid_x()) {
      raise blueprint_operation_error(operation.id(), ArtifactEntryPoint)
    }
    if variant.maximum_token_rows() > profile.max_query_tokens() ||
      dimensions.grid_y() != 1 ||
      dimensions.grid_z() != 1 ||
      dimensions.block_y() != 1 ||
      dimensions.block_z() != 1 {
      raise blueprint_operation_error(operation.id(), ArtifactEntryPoint)
    }
  }
  guard artifacts.modules().get(module_index) is Some(module_image) else {
    raise blueprint_operation_error(operation.id(), ArtifactModule)
  }
  if entry.identity() != contract.entry_point() ||
    module_image.digest() != contract.entry_point().family().module_digest() {
    raise blueprint_operation_error(operation.id(), ArtifactModule)
  }
  {
    operation_id: operation.id(),
    operation_kind: operation.kind(),
    kv_layer: layer,
    capability: operation.required_capability(),
    semantic_version: contract.semantic_version(),
    entry_point: contract.entry_point(),
    artifact_module_index: module_index,
    function_symbol: entry.function_symbol(),
    row_variants: entry.row_variants(),
    companion_dimensions: entry.companion_dimensions(),
    dimensions: contract.dimensions(),
    operands: ReadOnlyArray::from_array(operands),
  }
}

///|
/// Admits a pure startup blueprint for positioned rotary and paged attention.
///
/// All inputs are already-inert plans or admitted metadata. The result stores
/// only immutable identities, semantic roles, launch metadata, and scalar
/// offsets into separately owned startup arenas. It opens no native resource
/// and exposes no allocation or kernel-argument alias.
pub fn admit_paged_kv_execution_blueprint(
  model_plan : @plan.ModelPlan,
  limits : DeviceStepLimits,
  layout : @device_layout.DeviceKvLayout,
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
    KvSubsequence,
  )
  let profile = selected_paged_profile(contracts, profile_id, limits)
  let operations = paged_model_operations(model_plan)
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
        None,
        None,
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
    weight_arena_bytes: 0L,
    weight_alignment: 1L,
    activation_workspace_arena_bytes: memory_plan.arena_bytes(),
    graph_memory: derive_graph_memory_evidence(None),
    graph_shape: None,
    steps: ReadOnlyArray::from_array(steps),
  }
}
