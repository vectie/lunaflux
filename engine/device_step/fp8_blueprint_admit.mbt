///|
fn fp8_target_matches_device(
  target : @catalog.DeviceTarget,
  capability : @device.DeviceNumericCapability,
) -> Bool {
  target.compute_major() == capability.compute_major() &&
  target.compute_minor() == capability.compute_minor() &&
  target.supports_bf16() == capability.supports_bf16() &&
  target.supports_cublas_lt() == capability.supports_cublas_lt()
}

///|
fn validate_fp8_blueprint_evidence_v3(
  authority : @fp8_release_authority.Fp8ReusablePagedEnvelopeAuthorityV3,
  limits : DeviceStepLimits,
  static_plan : @device_plan.StaticDevicePlan,
  memory_plan : @device_memory.DeviceMemoryPlan,
  contracts : @launch_contract.PagedV4LaunchContractSet,
) -> Unit raise DeviceStepError {
  let recipe = authority.recipe()
  let startup = recipe.startup()
  let plan = startup.model_plan()
  let numeric = startup.device_numeric_capability()
  let artifacts = startup.kernel_artifacts().artifacts()
  let layout = recipe.kv_layout()
  let profile = recipe.profile()
  ignore(validate_plan_binding(plan, layout, limits))
  if plan.kv_execution() != PagedKeyValue ||
    static_plan.identity() != plan.identity() ||
    memory_plan.identity() != plan.identity() ||
    contracts.model_identity() != plan.identity() ||
    artifacts.model_identity() != plan.identity() ||
    recipe.weight_inspection().identity() != plan.identity() {
    raise blueprint_error(Identity)
  }
  let schema = plan.numeric_binding().digest()
  if static_plan.numeric_schema_digest() != schema ||
    contracts.numeric_schema_digest() != schema ||
    static_plan.weight_layout().numeric_schema_digest() != Some(schema) ||
    recipe.weight_inspection().numeric_schema_digest() != schema ||
    recipe.weight_inspection().layout() != static_plan.weight_layout() {
    raise blueprint_error(NumericSchema)
  }
  if numeric.version() != @device.DeviceNumericCapabilityVersion::v1() ||
    numeric.feature() != FiniteFp8E4M3W8A8F32AccumulateBf16OutputV1 ||
    !fp8_target_matches_device(authority.target(), numeric) {
    raise blueprint_error(DeviceNumericCapability)
  }
  let target = authority.target()
  if contracts.device_target() != target ||
    static_plan.target() != target ||
    memory_plan.target() != target ||
    artifacts.device_target() != target {
    raise blueprint_error(Target)
  }
  if contracts.catalog_version() != @catalog.CatalogVersion::v4() ||
    static_plan.catalog_version() != @catalog.CatalogVersion::v4() ||
    artifacts.catalog_version() != @catalog.CatalogVersion::v4() ||
    contracts.profiles().length() != 1 ||
    contracts.profiles()[0] != profile ||
    profile.max_query_rows() != limits.max_rows() ||
    profile.max_query_tokens() != limits.max_tokens() ||
    profile.max_page_table_entries() != limits.max_pages() {
    raise blueprint_error(Profile)
  }
  if contracts.device_kv_layout() != layout ||
    !layout.is_canonical_for(plan.identity(), plan.kv_cache_geometry()) {
    raise blueprint_error(Layout)
  }
  let shape = plan.shape_constraints()
  validate_blueprint_memory_plan(
    plan,
    memory_plan,
    checked_blueprint_product(
      shape.max_batch_rows().to_int64(),
      shape.max_sequence_tokens().to_int64(),
    ),
  )
  if memory_plan.operation_count() != plan.operations().length() ||
    static_plan.operations().length() != plan.operations().length() ||
    static_plan.weight_layout().length() != plan.tensor_count() {
    raise blueprint_error(OperationCount)
  }
  for index, operation in plan.operations() {
    let numeric_operation = plan.numeric_binding().operations()[index]
    let bound = static_plan.operations()[index]
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
fn fp8_projection_kind(kind : @plan.OperationKind) -> Bool {
  kind is (QkvProjection | OutputProjection | GatedMlp | LanguageModelHead)
}

///|
fn fp8_rebind_workspace_v3(
  step : I8PagedExecutionStep,
  launch : @fp8_runtime_recipe.Fp8ReusablePagedLaunchV3,
  workspace_base : Int64,
  arena_bytes : Int64,
) -> ReadOnlyArray[PagedKvOperandBinding] raise DeviceStepError {
  let result : Array[PagedKvOperandBinding] = []
  let mut workspace_count = 0
  for index, binding in step.operands {
    if binding.role == Workspace {
      if index != step.operands.length() - 1 ||
        binding.byte_count != launch.workspace_bytes() ||
        binding.alignment != launch.workspace_alignment() {
        raise blueprint_operation_error(step.operation_id, WorkspaceRegion)
      }
      let offset = workspace_base + launch.workspace_offset()
      if offset < workspace_base ||
        launch.workspace_bytes() > arena_bytes - offset ||
        offset % launch.workspace_alignment() != 0L {
        raise blueprint_operation_error(step.operation_id, WorkspaceRegion)
      }
      result.push({
        role: Workspace,
        space: ActivationWorkspaceArena,
        offset_bytes: offset,
        byte_count: launch.workspace_bytes(),
        alignment: launch.workspace_alignment(),
      })
      workspace_count += 1
    } else {
      result.push(binding)
    }
  }
  if workspace_count != 1 {
    raise blueprint_operation_error(step.operation_id, WorkspaceRegion)
  }
  ReadOnlyArray::from_array(result)
}

///|
pub fn admit_fp8_paged_graph_execution_blueprint_v3(
  authority : @fp8_release_authority.Fp8ReusablePagedEnvelopeAuthorityV3,
  limits : DeviceStepLimits,
  static_plan : @device_plan.StaticDevicePlan,
  memory_plan : @device_memory.DeviceMemoryPlan,
  contracts : @launch_contract.PagedV4LaunchContractSet,
) -> Fp8PagedExecutionBlueprintV3 raise DeviceStepError {
  validate_fp8_blueprint_evidence_v3(
    authority, limits, static_plan, memory_plan, contracts,
  )
  let recipe = authority.recipe()
  let plan = recipe.startup().model_plan()
  let profile = recipe.profile()
  let layout = recipe.kv_layout()
  let artifacts = recipe.startup().kernel_artifacts().artifacts()
  let workspace_base = memory_plan.arena_bytes()
  let fp8_workspace_bytes = recipe.workspace_capacity_bytes()
  if workspace_base < 0L ||
    fp8_workspace_bytes <= 0L ||
    fp8_workspace_bytes > 2147483647L - workspace_base {
    raise blueprint_error(WorkspaceRegion)
  }
  let arena_bytes = workspace_base + fp8_workspace_bytes
  let selected : Array[@launch_contract.PagedV4AotLaunchContract] = []
  for contract in contracts.contracts() {
    if contract.profile().id() == profile.id() {
      selected.push(contract)
    }
  }
  if selected.length() != plan.operations().length() {
    raise blueprint_error(OperationCount)
  }
  let launches = recipe.launches()
  let release_operations = authority.operations()
  if launches.length() != release_operations.length() {
    raise blueprint_error(OperationCount)
  }
  let steps : Array[Fp8PagedExecutionStepV3] = []
  let mut fp8_index = 0
  for index, operation in plan.operations() {
    if selected[index].operation_id() != operation.id() {
      raise blueprint_operation_error(operation.id(), OperationOrder)
    }
    let base = build_i8_v4_step(
      plan,
      operation,
      static_plan.operations()[index],
      selected[index],
      profile,
      layout,
      memory_plan,
      static_plan.weight_layout(),
      artifacts,
    )
    let is_fp8 = fp8_index < launches.length() &&
      launches[fp8_index].operation() == operation.id()
    let operands = if is_fp8 {
      let launch = launches[fp8_index]
      let release = release_operations[fp8_index]
      if !fp8_projection_kind(operation.kind()) ||
        release.operation() != operation.id() ||
        release.entry_point() != base.entry_point ||
        release.function_symbol() != base.function_symbol ||
        release.dimensions() != base.dimensions ||
        launch.entry_point() != base.entry_point ||
        launch.dimensions() != base.dimensions ||
        launch.operands() != selected[index].operands() ||
        release.operands() != selected[index].operands() {
        raise blueprint_operation_error(operation.id(), ContractSemantics)
      }
      fp8_index += 1
      fp8_rebind_workspace_v3(base, launch, workspace_base, arena_bytes)
    } else {
      if fp8_projection_kind(operation.kind()) {
        raise blueprint_operation_error(operation.id(), OperationNumericDigest)
      }
      base.operands
    }
    steps.push({
      operation_id: base.operation_id,
      operation_kind: base.operation_kind,
      operation_execution_digest: base.operation_execution_digest,
      entry_point: base.entry_point,
      artifact_module_index: base.artifact_module_index,
      function_symbol: base.function_symbol,
      dimensions: base.dimensions,
      operands,
      fp8_release_operation: is_fp8,
    })
  }
  if fp8_index != launches.length() {
    raise blueprint_error(OperationCount)
  }
  {
    authority,
    profile,
    limits,
    layout,
    weight_layout: static_plan.weight_layout(),
    memory_plan,
    artifacts,
    fp8_workspace_base: workspace_base,
    fp8_workspace_bytes,
    activation_workspace_arena_bytes: arena_bytes,
    steps: ReadOnlyArray::from_array(steps),
  }
}
