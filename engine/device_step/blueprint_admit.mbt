///|
fn blueprint_error(rule : PagedKvBlueprintRule) -> DeviceStepError {
  InvalidBlueprint(rule)
}

///|
fn blueprint_operation_error(
  operation : @plan.OperationId,
  rule : PagedKvBlueprintRule,
) -> DeviceStepError {
  InvalidBlueprintOperation(operation, rule)
}

///|
fn selected_paged_profile(
  contracts : @launch_contract.PagedKvLaunchContractSet,
  profile_id : @launch_contract.KernelProfileId,
  limits : DeviceStepLimits,
) -> @launch_contract.PagedKernelProfile raise DeviceStepError {
  for profile in contracts.profiles() {
    if profile.id() == profile_id {
      if profile.max_query_rows() != limits.max_rows() ||
        profile.max_query_tokens() != limits.max_tokens() ||
        profile.max_page_table_entries() != limits.max_pages() {
        raise blueprint_error(Profile)
      }
      return profile
    }
  }
  raise blueprint_error(Profile)
}

///|
fn validate_blueprint_evidence(
  model_plan : @plan.ModelPlan,
  limits : DeviceStepLimits,
  layout : @device_layout.DeviceKvLayout,
  memory_plan : @device_memory.DeviceMemoryPlan,
  contracts : @launch_contract.PagedKvLaunchContractSet,
  artifacts : @artifact.KernelArtifactBundle,
  expected_scope : @launch_contract.PagedContractScope,
) -> Unit raise DeviceStepError {
  ignore(validate_plan_binding(model_plan, layout, limits))
  let identity = model_plan.identity()
  if memory_plan.identity() != identity ||
    contracts.model_identity() != identity ||
    artifacts.model_identity() != identity {
    raise blueprint_error(Identity)
  }
  if contracts.device_kv_layout() != layout {
    raise blueprint_error(Layout)
  }
  let target = contracts.device_target()
  if memory_plan.target() != target || artifacts.device_target() != target {
    raise blueprint_error(Target)
  }
  let catalog = @catalog.CatalogVersion::v3()
  if contracts.catalog_version() != catalog ||
    artifacts.catalog_version() != catalog {
    raise blueprint_error(CatalogVersion)
  }
  if contracts.scope() != expected_scope {
    raise blueprint_error(ContractSemantics)
  }
  if memory_plan.operation_count() != model_plan.operations().length() ||
    memory_plan.token_rows() <= 0L ||
    memory_plan.arena_bytes() <= 0L {
    raise blueprint_error(OperationCount)
  }
  validate_blueprint_memory_plan(model_plan, memory_plan)
  let workspace = memory_plan.workspace()
  if workspace.offset_bytes() < memory_plan.activation_bytes() ||
    workspace.end_offset_bytes() > memory_plan.arena_bytes() {
    raise blueprint_error(WorkspaceRegion)
  }
}

///|
fn paged_model_operations(
  model_plan : @plan.ModelPlan,
) -> Array[@plan.PlanOperation] raise DeviceStepError {
  let result : Array[@plan.PlanOperation] = []
  let mut next_layer = 0
  let mut awaiting_attention = false
  for operation in model_plan.operations() {
    match operation.kind() {
      RotaryEmbedding => {
        guard operation.kv_layer() is Some(layer) &&
          layer.as_int() == next_layer &&
          !awaiting_attention else {
          raise blueprint_operation_error(operation.id(), OperationOrder)
        }
        awaiting_attention = true
        result.push(operation)
      }
      CausalAttention => {
        guard operation.kv_layer() is Some(layer) &&
          layer.as_int() == next_layer &&
          awaiting_attention else {
          raise blueprint_operation_error(operation.id(), OperationOrder)
        }
        awaiting_attention = false
        next_layer += 1
        result.push(operation)
      }
      _ => ()
    }
  }
  if awaiting_attention ||
    next_layer != model_plan.kv_cache_geometry().layer_count() ||
    result.length() != next_layer * 2 {
    raise blueprint_error(OperationOrder)
  }
  result
}

///|
fn selected_paged_contracts(
  contracts : @launch_contract.PagedKvLaunchContractSet,
  profile : @launch_contract.PagedKernelProfile,
) -> Array[@launch_contract.PagedKvAotLaunchContract] {
  let result : Array[@launch_contract.PagedKvAotLaunchContract] = []
  for contract in contracts.contracts() {
    if contract.profile().id() == profile.id() {
      result.push(contract)
    }
  }
  result
}

///|
fn expected_paged_roles(
  operation : @plan.PlanOperation,
  workspace_bytes : Int64,
) -> Array[@launch_contract.AotOperandRole] {
  let roles : Array[@launch_contract.AotOperandRole] = []
  for runtime in operation.runtime_inputs() {
    roles.push(RuntimeInput(runtime))
  }
  for input in operation.inputs() {
    roles.push(
      match input {
        ModelInput(reference) => ExternalInput(reference)
        Tensor(reference) => WeightInput(reference)
        Activation(reference) => ActivationInput(reference)
      },
    )
  }
  for output in operation.outputs() {
    roles.push(ActivationOutput(output))
  }
  match operation.persistent_state() {
    NoPersistentState => ()
    KvCacheReadWrite(layer) => {
      roles.push(PersistentKvComponent(layer, Key))
      roles.push(PersistentKvComponent(layer, Value))
    }
  }
  if workspace_bytes > 0L {
    roles.push(Workspace)
  }
  roles
}

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
