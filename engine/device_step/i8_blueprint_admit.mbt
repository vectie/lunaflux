///|
fn selected_i8_v4_profile(
  contracts : @launch_contract.PagedV4LaunchContractSet,
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
fn i8_target_matches_device(
  target : @catalog.DeviceTarget,
  capability : @device.DeviceNumericCapability,
) -> Bool {
  target.compute_major() == capability.compute_major() &&
  target.compute_minor() == capability.compute_minor() &&
  target.supports_bf16() == capability.supports_bf16() &&
  target.supports_cublas_lt() == capability.supports_cublas_lt()
}

///|
fn validate_i8_blueprint_evidence(
  model_plan : @plan.ModelPlan,
  device_numeric : @device.DeviceNumericCapability,
  policy : @i8_admission.I8ProductionPolicyAdmission,
  limits : DeviceStepLimits,
  layout : @device_layout.DeviceKvLayout,
  static_plan : @device_plan.StaticDevicePlan,
  memory_plan : @device_memory.DeviceMemoryPlan,
  contracts : @launch_contract.PagedV4LaunchContractSet,
  artifacts : @artifact.KernelArtifactBundle,
) -> Unit raise DeviceStepError {
  ignore(validate_plan_binding(model_plan, layout, limits))
  let identity = model_plan.identity()
  let numeric = model_plan.numeric_binding().digest()
  if model_plan.kv_execution() != PagedKeyValue ||
    policy.model_identity() != identity ||
    static_plan.identity() != identity ||
    memory_plan.identity() != identity ||
    contracts.model_identity() != identity ||
    artifacts.model_identity() != identity {
    raise blueprint_error(Identity)
  }
  if policy.numeric_schema_digest() != numeric ||
    static_plan.numeric_schema_digest() != numeric ||
    contracts.numeric_schema_digest() != numeric ||
    static_plan.weight_layout().numeric_schema_digest() != Some(numeric) {
    raise blueprint_error(NumericSchema)
  }
  if device_numeric.version() != @device.DeviceNumericCapabilityVersion::v1() ||
    device_numeric.feature() !=
    SymmetricI8WeightOnlyPerOutputChannelF32ScaleF32AccumulateBf16OutputV1 ||
    policy.device_numeric_digest() != device_numeric.digest() {
    raise blueprint_error(DeviceNumericCapability)
  }
  let target = contracts.device_target()
  if !i8_target_matches_device(target, device_numeric) ||
    static_plan.target() != target ||
    memory_plan.target() != target ||
    artifacts.device_target() != target {
    raise blueprint_error(Target)
  }
  let v4 = @catalog.CatalogVersion::v4()
  if contracts.catalog_version() != v4 ||
    static_plan.catalog_version() != v4 ||
    artifacts.catalog_version() != v4 ||
    policy.catalog_admission().catalog_version() != v4 {
    raise blueprint_error(CatalogVersion)
  }
  if contracts.device_kv_layout() != layout ||
    !layout.is_canonical_for(identity, model_plan.kv_cache_geometry()) {
    raise blueprint_error(Layout)
  }
  if memory_plan.operation_count() != model_plan.operations().length() ||
    memory_plan.token_rows() <= 0L ||
    memory_plan.arena_bytes() <= 0L {
    raise blueprint_error(OperationCount)
  }
  validate_blueprint_memory_plan(model_plan, memory_plan)
  let semantic = model_plan.operations()
  let physical = static_plan.operations()
  if semantic.length() != physical.length() ||
    static_plan.weight_layout().length() != model_plan.tensor_count() {
    raise blueprint_error(OperationCount)
  }
  for index, operation in semantic {
    let numeric_operation = model_plan.numeric_binding().operations()[index]
    let bound = physical[index]
    if numeric_operation.operation() != operation.id() ||
      bound.id() != operation.id() ||
      bound.operation_execution_digest() !=
      numeric_operation.execution().digest() ||
      bound.binding().operation_execution_digest() !=
      Some(numeric_operation.execution().digest()) ||
      bound.binding().aot_entry_point() is None ||
      bound.binding().semantic_version() != @catalog.KernelSemanticVersion::v4() {
      raise blueprint_operation_error(operation.id(), OperationNumericDigest)
    }
  }
}

///|
fn expected_i8_v4_roles(
  model_plan : @plan.ModelPlan,
  operation : @plan.PlanOperation,
  workspace_bytes : Int64,
) -> Array[@launch_contract.AotOperandRole] raise DeviceStepError {
  let roles : Array[@launch_contract.AotOperandRole] = []
  for runtime in operation.runtime_inputs() {
    roles.push(RuntimeInput(runtime))
  }
  for input in operation.inputs() {
    match input {
      ModelInput(reference) => roles.push(ExternalInput(reference))
      Activation(reference) => roles.push(ActivationInput(reference))
      Tensor(reference) => {
        guard model_plan.tensor(reference) is Some(tensor) &&
          tensor.role() == ParameterTensor else {
          raise blueprint_operation_error(operation.id(), OperandRole)
        }
        roles.push(WeightInput(reference))
        match tensor.scale() {
          Some(scale) => roles.push(WeightScaleInput(reference, scale))
          None => ()
        }
      }
    }
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
fn bind_i8_layout_region(
  operation : @plan.OperationId,
  role : @launch_contract.AotOperandRole,
  reference : @plan.TensorRef,
  operand : @launch_contract.AotOperand,
  weight_layout : @device_materialize.DeviceWeightLayout,
) -> PagedKvOperandBinding raise DeviceStepError {
  guard weight_layout.region(reference) is Some(region) else {
    raise blueprint_operation_error(operation, ScaleRegion)
  }
  if operand.byte_count() != region.byte_length() ||
    region.end_offset_bytes() > weight_layout.arena_bytes() {
    raise blueprint_operation_error(operation, OperandSize)
  }
  if operand.alignment() > weight_layout.alignment_bytes() ||
    weight_layout.alignment_bytes() % operand.alignment() != 0L ||
    region.offset_bytes() % operand.alignment() != 0L {
    raise blueprint_operation_error(operation, OperandAlignment)
  }
  {
    role,
    space: WeightArena,
    offset_bytes: region.offset_bytes(),
    byte_count: region.byte_length(),
    alignment: operand.alignment(),
  }
}

///|
fn bind_i8_v4_operand(
  model_plan : @plan.ModelPlan,
  operation : @plan.PlanOperation,
  operand : @launch_contract.AotOperand,
  profile : @launch_contract.PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
  memory_plan : @device_memory.DeviceMemoryPlan,
  static_operation : @device_plan.StaticOperation,
  weight_layout : @device_materialize.DeviceWeightLayout,
) -> PagedKvOperandBinding raise DeviceStepError {
  match operand.role() {
    RuntimeInput(input) => {
      if operand.byte_count() != runtime_operand_bytes(input, profile) ||
        operand.alignment() != 4L {
        raise blueprint_operation_error(operation.id(), OperandSize)
      }
      {
        role: operand.role(),
        space: StepDescriptor(runtime_stage(input)),
        offset_bytes: 0L,
        byte_count: operand.byte_count(),
        alignment: operand.alignment(),
      }
    }
    ExternalInput(reference) =>
      bind_external_operand(
        operation.id(),
        reference,
        operand,
        profile,
        Some(static_operation),
      )
    WeightInput(reference) => {
      guard weight_layout.region(reference) is Some(region) else {
        raise blueprint_operation_error(operation.id(), OperandRole)
      }
      if !static_input_matches(static_operation, WeightRegion(region)) {
        raise blueprint_operation_error(operation.id(), OperandRole)
      }
      bind_i8_layout_region(
        operation.id(),
        operand.role(),
        reference,
        operand,
        weight_layout,
      )
    }
    WeightScaleInput(weight, scale) => {
      guard model_plan.tensor(weight) is Some(parameter) &&
        parameter.role() == ParameterTensor &&
        parameter.scale() == Some(scale) &&
        model_plan.tensor(scale.tensor()) is Some(metadata) &&
        metadata.role() == ScaleMetadataTensor else {
        raise blueprint_operation_error(operation.id(), ScaleRegion)
      }
      bind_i8_layout_region(
        operation.id(),
        operand.role(),
        scale.tensor(),
        operand,
        weight_layout,
      )
    }
    ActivationInput(reference) =>
      bind_activation_operand(
        operation.id(),
        reference,
        false,
        operand,
        profile,
        memory_plan,
      )
    ActivationOutput(reference) =>
      bind_activation_operand(
        operation.id(),
        reference,
        true,
        operand,
        profile,
        memory_plan,
      )
    PersistentKvComponent(layer, component) =>
      bind_kv_operand(operation.id(), layer, component, operand, layout)
    Workspace => bind_workspace_operand(operation.id(), operand, memory_plan)
  }
}

///|
fn build_i8_v4_step(
  model_plan : @plan.ModelPlan,
  operation : @plan.PlanOperation,
  static_operation : @device_plan.StaticOperation,
  contract : @launch_contract.PagedV4AotLaunchContract,
  profile : @launch_contract.PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
  memory_plan : @device_memory.DeviceMemoryPlan,
  weight_layout : @device_materialize.DeviceWeightLayout,
  artifacts : @artifact.KernelArtifactBundle,
) -> I8PagedExecutionStep raise DeviceStepError {
  let digest = model_plan.numeric_binding().operations()[operation.id().as_int()]
    .execution()
    .digest()
  if contract.operation_id() != operation.id() ||
    contract.operation_kind() != operation.kind() ||
    contract.operation_shape() != operation.shape() ||
    contract.capability() != operation.required_capability() ||
    contract.semantic_version() != @catalog.KernelSemanticVersion::v4() ||
    contract.operation_execution_digest() != digest ||
    contract.kv_layer() != operation.kv_layer() ||
    contract.runtime_inputs() != operation.runtime_inputs() ||
    contract.persistent_state() != operation.persistent_state() ||
    contract.kv_layout_version() != layout.stable_layout_version() ||
    contract.tokens_per_page() != layout.tokens_per_page() {
    raise blueprint_operation_error(operation.id(), ContractSemantics)
  }
  let expected = expected_i8_v4_roles(
    model_plan,
    operation,
    contract.workspace_bytes(),
  )
  if contract.operands().length() != expected.length() {
    raise blueprint_operation_error(operation.id(), OperandCount)
  }
  let operands : Array[PagedKvOperandBinding] = []
  for index, operand in contract.operands() {
    if operand.role() != expected[index] {
      raise blueprint_operation_error(operation.id(), OperandRole)
    }
    operands.push(
      bind_i8_v4_operand(
        model_plan, operation, operand, profile, layout, memory_plan, static_operation,
        weight_layout,
      ),
    )
  }
  guard artifacts.entry_point(contract.entry_point()) is Some(entry) &&
    artifacts.modules().get(entry.module_index()) is Some(module_image) &&
    module_image.digest() == contract.entry_point().family().module_digest() else {
    raise blueprint_operation_error(operation.id(), ArtifactModule)
  }
  {
    operation_id: operation.id(),
    operation_kind: operation.kind(),
    kv_layer: operation.kv_layer(),
    capability: operation.required_capability(),
    semantic_version: contract.semantic_version(),
    operation_execution_digest: digest,
    entry_point: contract.entry_point(),
    artifact_module_index: entry.module_index(),
    function_symbol: entry.function_symbol(),
    dimensions: contract.dimensions(),
    operands: ReadOnlyArray::from_array(operands),
  }
}

///|
/// Admits the exact catalog-v4 complete paged graph for the first production
/// symmetric-I8 policy. This is inert pre-hardware evidence only. In
/// particular, it cannot construct an executor until a separately
/// authenticated numeric-weight loader authority is joined by a later owner.
pub fn admit_i8_paged_graph_execution_blueprint_v1(
  model_plan : @plan.ModelPlan,
  device_numeric : @device.DeviceNumericCapability,
  policy : @i8_admission.I8ProductionPolicyAdmission,
  limits : DeviceStepLimits,
  layout : @device_layout.DeviceKvLayout,
  static_plan : @device_plan.StaticDevicePlan,
  memory_plan : @device_memory.DeviceMemoryPlan,
  profile_id : @launch_contract.KernelProfileId,
  contracts : @launch_contract.PagedV4LaunchContractSet,
  artifacts : @artifact.KernelArtifactBundle,
) -> I8PagedExecutionBlueprint raise DeviceStepError {
  // Production policy v1 is deliberately a single reviewed launch profile.
  // Keeping unselected profile artifacts would create code-loading authority
  // that is not represented by the retained semantic step sequence.
  if contracts.profiles().length() != 1 ||
    contracts.profiles()[0].id() != profile_id {
    raise blueprint_error(Profile)
  }
  validate_i8_blueprint_evidence(
    model_plan, device_numeric, policy, limits, layout, static_plan, memory_plan,
    contracts, artifacts,
  )
  let profile = selected_i8_v4_profile(contracts, profile_id, limits)
  let operations = model_plan.operations()
  let selected : Array[@launch_contract.PagedV4AotLaunchContract] = []
  for contract in contracts.contracts() {
    if contract.profile().id() == profile.id() {
      selected.push(contract)
    }
  }
  if selected.length() != operations.length() {
    raise blueprint_error(OperationCount)
  }
  let steps : Array[I8PagedExecutionStep] = []
  for index, operation in operations {
    if selected[index].operation_id() != operation.id() {
      raise blueprint_operation_error(operation.id(), OperationOrder)
    }
    steps.push(
      build_i8_v4_step(
        model_plan,
        operation,
        static_plan.operations()[index],
        selected[index],
        profile,
        layout,
        memory_plan,
        static_plan.weight_layout(),
        artifacts,
      ),
    )
  }
  {
    identity: model_plan.identity(),
    numeric_schema_digest: model_plan.numeric_binding().digest(),
    i8_admission_digest: policy.digest(),
    device_numeric_digest: device_numeric.digest(),
    target: contracts.device_target(),
    catalog_version: contracts.catalog_version(),
    profile,
    limits,
    layout,
    weight_layout: static_plan.weight_layout(),
    memory_plan,
    artifacts,
    activation_workspace_arena_bytes: memory_plan.arena_bytes(),
    steps: ReadOnlyArray::from_array(steps),
  }
}
