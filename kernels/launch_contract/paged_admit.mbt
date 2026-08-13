///|
const PAGED_V2_STEP_COUNT_BYTES : Int64 = 20L

///|
const PAGED_V2_INT32_ALIGNMENT : Int64 = 4L

///|
fn paged_profile_is_known(
  profiles : ArrayView[PagedKernelProfile],
  profile_id : KernelProfileId,
) -> Bool {
  profiles.any(profile => profile.id == profile_id)
}

///|
fn paged_profile_for(
  profiles : ArrayView[PagedKernelProfile],
  profile_id : KernelProfileId,
) -> PagedKernelProfile? {
  for profile in profiles {
    if profile.id == profile_id {
      return Some(profile)
    }
  }
  None
}

///|
fn validate_paged_profiles(
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
    if profile.max_query_rows > row_limit ||
      profile.max_query_tokens.to_int64() > token_limit ||
      profile.max_page_table_entries.to_int64() > page_reference_limit {
      raise InvalidProfile(OutsideModelEnvelope)
    }
    for earlier in 0..<index {
      if profiles[earlier].id == profile.id {
        raise DuplicateProfile(profile.id)
      }
    }
  }
}

///|
fn paged_runtime_bytes(
  role : @plan.OperationRuntimeInput,
  profile : PagedKernelProfile,
) -> Int64 {
  match role {
    StepCounts => PAGED_V2_STEP_COUNT_BYTES
    QueryPositions => profile.max_query_tokens.to_int64() * 4L
    QueryRowOffsets | PageTableOffsets =>
      (profile.max_query_rows.to_int64() + 1L) * 4L
    SequenceLengths => profile.max_query_rows.to_int64() * 4L
    PageTablePageIndices => profile.max_page_table_entries.to_int64() * 4L
  }
}

///|
fn canonical_paged_roles(
  operation : @plan.PlanOperation,
  binding : @catalog.KernelBinding,
) -> Array[AotOperandRole] {
  let roles : Array[AotOperandRole] = []
  for runtime_input in operation.runtime_inputs() {
    roles.push(RuntimeInput(runtime_input))
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
  if binding.workspace_bytes() > 0L {
    roles.push(Workspace)
  }
  roles
}

///|
fn validate_paged_operand_set(
  operation : @plan.PlanOperation,
  binding : @catalog.KernelBinding,
  profile : PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
  operands : ReadOnlyArray[AotOperand],
  maximum : Int,
) -> Unit raise LaunchContractError {
  if operands.is_empty() || operands.length() > maximum {
    raise OperandCountExceeded(operation=operation.id(), maximum~)
  }
  let expected = canonical_paged_roles(operation, binding)
  if operands.length() != expected.length() {
    raise InvalidOperand(operation.id(), MissingSemanticRole)
  }
  for index, operand in operands {
    if operand.role != expected[index] {
      raise InvalidOperand(operation.id(), PositionalRoleMismatch)
    }
    match operand.role {
      RuntimeInput(role) =>
        if operand.byte_count != paged_runtime_bytes(role, profile) ||
          operand.alignment != PAGED_V2_INT32_ALIGNMENT {
          raise InvalidOperand(operation.id(), RuntimeInputSizeMismatch)
        }
      PersistentKvComponent(_, _) =>
        if operand.byte_count != layout.component_stride_bytes() ||
          operand.alignment != layout.segment_alignment_bytes() {
          raise InvalidOperand(operation.id(), PersistentKvMismatch)
        }
      Workspace =>
        if operand.byte_count != binding.workspace_bytes() ||
          operand.alignment != binding.workspace_alignment() {
          raise InvalidOperand(operation.id(), WorkspaceMismatch)
        }
      ExternalInput(_)
      | WeightInput(_)
      | ActivationInput(_)
      | ActivationOutput(_) => ()
    }
  }
}

///|
fn paged_input_for(
  inputs : ArrayView[PagedKvAotLaunchContractInput],
  profile_id : KernelProfileId,
  operation_id : @plan.OperationId,
) -> PagedKvAotLaunchContractInput? {
  for input in inputs {
    if input.profile_id == profile_id && input.operation_id == operation_id {
      return Some(input)
    }
  }
  None
}

///|
fn is_paged_kv_operation(operation : @plan.PlanOperation) -> Bool {
  operation.kind() is (RotaryEmbedding | CausalAttention)
}

///|
fn validate_paged_binding(
  operation : @plan.PlanOperation,
  binding : @catalog.KernelBinding,
  layout : @device_layout.DeviceKvLayout,
) -> @catalog.KernelExecutionSemantics raise LaunchContractError {
  if binding.operation_id() != operation.id() ||
    binding.capability() != operation.required_capability() {
    raise BindingOrderMismatch(operation.id())
  }
  if binding.semantic_version() != @catalog.KernelSemanticVersion::v2() {
    raise CatalogVersionMismatch
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
  semantics
}

///|
/// Admits only positioned-rotary and paged-attention AOT metadata.
///
/// Operand positions are part of semantic version 2. `StepCounts` is exactly
/// five little-endian Int32 values; every other runtime role is an exact Int32
/// capacity. Persistent KV is the exact ordered Key and Value component-base
/// pair for the semantic layer; total physical page count only sizes spans.
/// Contracts for every other model operation are rejected: this set does not
/// claim full-graph or worker-step execution support.
pub fn admit_paged_kv(
  model_plan : @plan.ModelPlan,
  resolved : @catalog.ResolvedKernelCatalog,
  device_kv_layout : @device_layout.DeviceKvLayout,
  profiles : ArrayView[PagedKernelProfile],
  inputs : ArrayView[PagedKvAotLaunchContractInput],
  limits : LaunchContractLimits,
) -> PagedKvLaunchContractSet raise LaunchContractError {
  if model_plan.kv_execution() != PagedKeyValue {
    raise UnsupportedExecutionContract
  }
  if resolved.catalog_version() != @catalog.CatalogVersion::v2() {
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
  validate_paged_profiles(
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
  for index, operation in operations {
    ignore(validate_paged_binding(operation, bindings[index], device_kv_layout))
  }
  for index, input in inputs {
    if !paged_profile_is_known(profiles, input.profile_id) {
      raise UnknownProfile(input.profile_id)
    }
    guard operation_index(operations, input.operation_id) is Some(op_index) else {
      raise BindingOrderMismatch(input.operation_id)
    }
    if !is_paged_kv_operation(operations[op_index]) {
      raise ContractForNonPagedKvOperation(input.profile_id, input.operation_id)
    }
    let binding = bindings[op_index]
    match binding.implementation() {
      VendorBf16GemmImplicitHeuristicV1 =>
        raise ContractForVendorKernel(input.profile_id, input.operation_id)
      ContentAddressedAot(family) =>
        if input.entry_point.family() != family {
          raise KernelFamilyMismatch(input.profile_id, input.operation_id)
        }
    }
    for earlier in 0..<index {
      if inputs[earlier].profile_id == input.profile_id &&
        inputs[earlier].operation_id == input.operation_id {
        raise DuplicateContract(input.profile_id, input.operation_id)
      }
    }
    guard paged_profile_for(profiles, input.profile_id) is Some(profile) else {
      raise UnknownProfile(input.profile_id)
    }
    validate_paged_operand_set(
      operations[op_index],
      binding,
      profile,
      device_kv_layout,
      input.operands,
      limits.max_operands_per_contract,
    )
  }
  let admitted : Array[PagedKvAotLaunchContract] = []
  for profile in profiles {
    for index, operation in operations {
      if !is_paged_kv_operation(operation) {
        continue
      }
      let binding = bindings[index]
      match binding.implementation() {
        VendorBf16GemmImplicitHeuristicV1 => ()
        ContentAddressedAot(_) => {
          guard paged_input_for(inputs, profile.id, operation.id())
            is Some(input) else {
            raise MissingContract(profile.id, operation.id())
          }
          let semantics = validate_paged_binding(
            operation, binding, device_kv_layout,
          )
          admitted.push({
            profile,
            operation_id: operation.id(),
            operation_kind: operation.kind(),
            operation_shape: operation.shape(),
            capability: operation.required_capability(),
            semantic_version: binding.semantic_version(),
            kv_layer: semantics.kv_layer(),
            runtime_inputs: semantics.runtime_inputs(),
            persistent_state: semantics.persistent_state(),
            kv_layout_version: device_kv_layout.stable_layout_version(),
            tokens_per_page: device_kv_layout.tokens_per_page(),
            entry_point: input.entry_point,
            dimensions: input.dimensions,
            operands: input.operands,
            workspace_bytes: binding.workspace_bytes(),
            workspace_alignment: binding.workspace_alignment(),
          })
        }
      }
    }
  }
  {
    model_identity: model_plan.identity(),
    device_target: resolved.device_target(),
    catalog_version: resolved.catalog_version(),
    device_kv_layout,
    profiles: ReadOnlyArray::from_array(profiles),
    contracts: ReadOnlyArray::from_array(admitted),
  }
}
