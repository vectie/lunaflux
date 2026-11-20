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
  let mut has_exact_profile = false
  for profile in contracts.profiles() {
    if profile.max_query_rows() == limits.max_rows() &&
      profile.max_query_tokens() == limits.max_tokens() &&
      profile.max_page_table_entries() == limits.max_pages() {
      has_exact_profile = true
    }
  }
  if !has_exact_profile {
    raise blueprint_error(Profile)
  }
  if memory_plan.operation_count() != model_plan.operations().length() ||
    memory_plan.token_rows() <= 0L ||
    memory_plan.arena_bytes() <= 0L {
    raise blueprint_error(OperationCount)
  }
  validate_blueprint_memory_plan(
    model_plan,
    memory_plan,
    limits.max_tokens().to_int64(),
  )
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
