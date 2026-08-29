///|
fn validate_paged_v4_binding(
  model_plan : @plan.ModelPlan,
  operation : @plan.PlanOperation,
  binding : @catalog.KernelBinding,
  layout : @device_layout.DeviceKvLayout,
) -> (
  @catalog.KernelExecutionSemantics,
  @numeric_contract.OperationExecutionDigest,
  @catalog.AotKernelEntryPoint,
) raise LaunchContractError {
  if binding.operation_id() != operation.id() ||
    binding.capability() != operation.required_capability() {
    raise BindingOrderMismatch(operation.id())
  }
  if binding.semantic_version() != @catalog.KernelSemanticVersion::v4() {
    raise CatalogVersionMismatch
  }
  let execution = paged_v4_execution(model_plan, operation.id())
  let digest = execution.digest()
  if binding.operation_execution_digest() != Some(digest) {
    raise NumericBindingMismatch(operation.id())
  }
  guard binding.execution_semantics() is Some(semantics) else {
    raise CatalogVersionMismatch
  }
  if semantics.kv_layer() != operation.kv_layer() ||
    semantics.runtime_inputs() != operation.runtime_inputs() ||
    semantics.persistent_state() != operation.persistent_state() {
    raise BindingOrderMismatch(operation.id())
  }
  match semantics.kv_layout() {
    Some(semantic_layout) =>
      if semantic_layout.layout() != layout.geometry().layout() ||
        semantic_layout.stable_version() != layout.stable_layout_version() ||
        semantic_layout.tokens_per_page() != layout.tokens_per_page() {
        raise DeviceKvLayoutMismatch
      }
    None =>
      if operation.persistent_state() != NoPersistentState {
        raise DeviceKvLayoutMismatch
      }
  }
  guard binding.aot_entry_point() is Some(entry_point) else {
    raise NumericBindingMismatch(operation.id())
  }
  match binding.implementation() {
    VendorBf16GemmImplicitHeuristicV1 =>
      raise NumericBindingMismatch(operation.id())
    ContentAddressedAot(family) =>
      if entry_point.family() != family {
        raise NumericBindingMismatch(operation.id())
      }
  }
  (semantics, digest, entry_point)
}

///|
fn validate_paged_v4_profiles(
  profiles : ArrayView[PagedKernelProfile],
  constraints : @plan.ShapeConstraints,
  layout : @device_layout.DeviceKvLayout,
  limits : LaunchContractLimits,
) -> Unit raise LaunchContractError {
  if profiles.is_empty() || profiles.length() > limits.max_profiles {
    raise ProfileCountExceeded(maximum=limits.max_profiles)
  }
  let row_limit = constraints.max_batch_rows()
  let sequence_limit = constraints.max_sequence_tokens()
  let token_limit = row_limit.to_int64() * sequence_limit.to_int64()
  let page_span = (
      sequence_limit.to_int64() + layout.tokens_per_page().to_int64() - 1L
    ) /
    layout.tokens_per_page().to_int64()
  let page_reference_limit = row_limit.to_int64() * page_span
  for index, profile in profiles {
    if profile.id.as_int() != index + 1 ||
      profile.max_query_rows > row_limit ||
      profile.max_query_tokens.to_int64() > token_limit ||
      profile.max_page_table_entries.to_int64() > page_reference_limit {
      raise InvalidProfile(OutsideModelEnvelope)
    }
  }
}

///|
/// Admits one exact inert catalog-v4 AOT launch contract for every operation
/// and profile in a complete paged graph.
///
/// The operand sequence is derived from `ModelPlan`; callers cannot choose a
/// weight/scale association. Every schema-owned scale immediately follows its
/// weight. This function opens no artifact or module, allocates no device
/// resource, serializes no bootstrap, and launches no work.
pub fn admit_paged_graph_v4(
  model_plan : @plan.ModelPlan,
  resolved : @catalog.ResolvedKernelCatalog,
  device_kv_layout : @device_layout.DeviceKvLayout,
  profiles : ArrayView[PagedKernelProfile],
  inputs : ArrayView[PagedV4AotLaunchContractInput],
  limits : LaunchContractLimits,
) -> PagedV4LaunchContractSet raise LaunchContractError {
  if model_plan.kv_execution() != PagedKeyValue {
    raise UnsupportedExecutionContract
  }
  if resolved.catalog_version() != @catalog.CatalogVersion::v4() {
    raise CatalogVersionMismatch
  }
  if resolved.model_identity() != model_plan.identity() ||
    device_kv_layout.identity() != model_plan.identity() {
    raise IdentityMismatch
  }
  if device_kv_layout.geometry() != model_plan.kv_cache_geometry() ||
    device_kv_layout.stable_layout_version() !=
    model_plan.kv_cache_geometry().layout().stable_version() {
    raise DeviceKvLayoutMismatch
  }
  validate_paged_v4_profiles(
    profiles,
    model_plan.shape_constraints(),
    device_kv_layout,
    limits,
  )
  if inputs.length() > limits.max_contracts {
    raise ContractCountExceeded(maximum=limits.max_contracts)
  }
  let operations = model_plan.operations()
  let bindings = resolved.bindings()
  if operations.length() != bindings.length() {
    raise BindingCountMismatch
  }
  if operations.is_empty() ||
    profiles.length() > limits.max_contracts / operations.length() {
    raise ContractCountExceeded(maximum=limits.max_contracts)
  }
  let expected_contracts = profiles.length() * operations.length()
  if inputs.length() != expected_contracts {
    raise ContractCountMismatch
  }
  for index, operation in operations {
    ignore(
      validate_paged_v4_binding(
        model_plan,
        operation,
        bindings[index],
        device_kv_layout,
      ),
    )
  }
  let admitted : Array[PagedV4AotLaunchContract] = []
  for input_index, input in inputs {
    let profile_index = input_index / operations.length()
    let op_index = input_index % operations.length()
    let profile = profiles[profile_index]
    let operation = operations[op_index]
    let binding = bindings[op_index]
    if input.profile_id != profile.id || input.operation_id != operation.id() {
      raise BindingOrderMismatch(input.operation_id)
    }
    let (semantics, digest, entry_point) = validate_paged_v4_binding(
      model_plan, operation, binding, device_kv_layout,
    )
    if input.entry_point != entry_point {
      raise EntryPointMismatch(input.profile_id, input.operation_id)
    }
    validate_paged_v4_operand_set(
      model_plan,
      operation,
      binding,
      profile,
      device_kv_layout,
      input.operands,
      limits.max_operands_per_contract,
    )
    admitted.push({
      profile,
      operation_id: operation.id(),
      operation_kind: operation.kind(),
      operation_shape: operation.shape(),
      capability: operation.required_capability(),
      semantic_version: binding.semantic_version(),
      operation_execution_digest: digest,
      kv_layer: semantics.kv_layer(),
      runtime_inputs: semantics.runtime_inputs(),
      persistent_state: semantics.persistent_state(),
      kv_layout_version: device_kv_layout.stable_layout_version(),
      tokens_per_page: device_kv_layout.tokens_per_page(),
      entry_point,
      dimensions: input.dimensions,
      operands: input.operands,
      workspace_bytes: binding.workspace_bytes(),
      workspace_alignment: binding.workspace_alignment(),
    })
  }
  {
    model_identity: model_plan.identity(),
    numeric_schema_digest: model_plan.numeric_binding().digest(),
    device_target: resolved.device_target(),
    catalog_version: resolved.catalog_version(),
    device_kv_layout,
    profiles: ReadOnlyArray::from_array(profiles),
    contracts: ReadOnlyArray::from_array(admitted),
  }
}
