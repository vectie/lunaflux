///|
fn invalid_operation(
  operation_id : @plan.OperationId,
  rule : DeviceExecutorOperationRule,
) -> DeviceExecutorError {
  InvalidOperation(operation_id, rule)
}

///|
fn checked_bf16_bytes(
  rows : Int64,
  width : Int64,
  operation_id : @plan.OperationId,
) -> Int64 raise DeviceExecutorError {
  let maximum = 9223372036854775807L
  if rows <= 0L || width <= 0L || rows > maximum / width {
    raise invalid_operation(operation_id, SizeOverflow)
  }
  let elements = rows * width
  if elements > maximum / 2L {
    raise invalid_operation(operation_id, SizeOverflow)
  }
  elements * 2L
}

///|
fn validate_top_level_evidence(
  static_plan : @device_plan.StaticDevicePlan,
  memory_plan : @device_memory.DeviceMemoryPlan,
  profile : @device_profile.DeviceExecutionProfile,
  weight_layout : @device_materialize.DeviceWeightLayout,
  weight_allocation_bytes : Int64,
  contracts : @launch_contract.LaunchContractSet,
  artifacts : @artifact.KernelArtifactBundle,
  context_target_matches : Bool,
) -> Unit raise DeviceExecutorError {
  if static_plan.catalog_version() != @catalog.CatalogVersion::v1() ||
    contracts.catalog_version() != @catalog.CatalogVersion::v1() ||
    artifacts.catalog_version() != @catalog.CatalogVersion::v1() {
    raise InvalidPlan(CatalogVersion)
  }
  let identity = static_plan.identity()
  if memory_plan.identity() != identity ||
    profile.identity() != identity ||
    weight_layout.identity().content() != identity.content() ||
    contracts.model_identity() != identity ||
    artifacts.model_identity() != identity {
    raise InvalidPlan(Identity)
  }
  let target = static_plan.target()
  if memory_plan.target() != target ||
    profile.target() != target ||
    contracts.device_target() != target ||
    artifacts.device_target() != target {
    raise InvalidPlan(Target)
  }
  if static_plan.catalog_version() != contracts.catalog_version() ||
    static_plan.catalog_version() != artifacts.catalog_version() {
    raise InvalidPlan(CatalogVersion)
  }
  if !context_target_matches {
    raise InvalidPlan(ContextCapability)
  }
  if weight_layout.arena_bytes() != static_plan.weight_arena_bytes() ||
    weight_layout.alignment_bytes() != static_plan.weight_alignment() ||
    weight_layout.length() != static_plan.weight_tensor_count() ||
    weight_allocation_bytes != weight_layout.arena_bytes() {
    raise InvalidPlan(WeightLayout)
  }
  if memory_plan.arena_bytes() <= 0L ||
    memory_plan.operation_count() != static_plan.operations().length() ||
    memory_plan.final_output() != static_plan.final_output() {
    raise InvalidPlan(MemoryPlan)
  }
  if profile.operation_count() != static_plan.operations().length() ||
    profile.token_staging_capacity_bytes() <= 0L ||
    profile.token_staging().logical_bytes() <= 0L ||
    profile.token_staging().logical_bytes() >
    profile.token_staging_capacity_bytes() ||
    profile.final_output().value().reference() != static_plan.final_output() {
    raise InvalidPlan(ExecutionProfile)
  }
}

///|
fn selected_kernel_profile(
  contracts : @launch_contract.LaunchContractSet,
  profile : @device_profile.DeviceExecutionProfile,
  profile_id : @launch_contract.KernelProfileId,
) -> @launch_contract.KernelProfile raise DeviceExecutorError {
  for candidate in contracts.profiles() {
    if candidate.id() == profile_id {
      let shape = profile.shape().kernel_shape()
      if candidate.batch_rows() != shape.batch_rows() ||
        candidate.sequence_tokens() != shape.sequence_tokens() ||
        candidate.token_rows() != shape.token_rows() {
        raise InvalidPlan(KernelProfile)
      }
      return candidate
    }
  }
  raise InvalidPlan(KernelProfile)
}

///|
fn input_region(
  operation_profile : @device_profile.OperationExecutionProfile,
  role : @launch_contract.AotOperandRole,
) -> @device_profile.ExecutionRegionView? {
  match role {
    ExternalInput(reference) => {
      for input in operation_profile.inputs() {
        if input is TokenIds(region) && reference == TokenIds {
          return Some(region)
        }
      }
      None
    }
    WeightInput(reference) => {
      for input in operation_profile.inputs() {
        if input is Weight(input_reference, region) &&
          input_reference == reference {
          return Some(region)
        }
      }
      None
    }
    WeightScaleInput(_, _) => None
    ActivationInput(reference) => {
      for input in operation_profile.inputs() {
        if input is Activation(value) && value.reference() == reference {
          return Some(value.region())
        }
      }
      None
    }
    ActivationOutput(reference) => {
      for output in operation_profile.outputs() {
        if output.reference() == reference {
          return Some(output.region())
        }
      }
      None
    }
    RuntimeInput(_) | PersistentKvComponent(_, _) => None
    Workspace => Some(operation_profile.workspace())
  }
}

///|
fn bind_operand(
  operation_id : @plan.OperationId,
  operation_profile : @device_profile.OperationExecutionProfile,
  operand : @launch_contract.AotOperand,
) -> RegionBinding raise DeviceExecutorError {
  guard input_region(operation_profile, operand.role()) is Some(region) else {
    raise invalid_operation(operation_id, OperandMissing)
  }
  if region.logical_bytes() != operand.byte_count() {
    raise invalid_operation(operation_id, OperandSize)
  }
  if region.alignment_bytes() != operand.alignment() {
    raise invalid_operation(operation_id, OperandAlignment)
  }
  if operand.role() is Workspace && region.logical_bytes() == 0L {
    raise invalid_operation(operation_id, Workspace)
  }
  {
    allocation_space: region.space(),
    offset_bytes: region.offset_bytes(),
    byte_count: region.logical_bytes(),
    alignment: region.alignment_bytes(),
  }
}

///|
fn function_blueprint_index(
  functions : Array[FunctionBlueprint],
  identity : @catalog.AotKernelEntryPoint,
) -> Int? {
  for index, function in functions {
    if function.identity == identity {
      return Some(index)
    }
  }
  None
}

///|
fn append_function_blueprint(
  artifacts : @artifact.KernelArtifactBundle,
  functions : Array[FunctionBlueprint],
  identity : @catalog.AotKernelEntryPoint,
) -> Int raise DeviceExecutorError {
  if function_blueprint_index(functions, identity) is Some(index) {
    return index
  }
  guard artifacts.entry_point(identity) is Some(entry) else {
    raise InvalidPlan(ArtifactEntryPoint)
  }
  let module_index = entry.module_index()
  guard artifacts.modules().get(module_index) is Some(module_image) else {
    raise InvalidPlan(ArtifactModule)
  }
  if module_image.digest() != identity.family().module_digest() {
    raise InvalidPlan(ArtifactModule)
  }
  functions.push({ identity, module_index, symbol: entry.function_symbol() })
  functions.length() - 1
}

///|
fn aot_step_blueprint(
  operation : @device_plan.StaticOperation,
  operation_profile : @device_profile.OperationExecutionProfile,
  contract : @launch_contract.AotLaunchContract,
  artifacts : @artifact.KernelArtifactBundle,
  functions : Array[FunctionBlueprint],
) -> AotStepBlueprint raise DeviceExecutorError {
  if contract.operation_id() != operation.id() ||
    contract.operation_kind() != operation.kind() ||
    contract.operation_shape() != operation.shape() ||
    contract.capability() != operation.binding().capability() {
    raise invalid_operation(operation.id(), ContractMismatch)
  }
  match operation.binding().implementation() {
    ContentAddressedAot(family) =>
      if contract.entry_point().family() != family {
        raise invalid_operation(operation.id(), ContractMismatch)
      }
    VendorBf16GemmImplicitHeuristicV1 =>
      raise invalid_operation(operation.id(), ContractMismatch)
  }
  let dimensions = @device.KernelLaunchDimensions::new(
    grid_x=contract.dimensions().grid_x(),
    grid_y=contract.dimensions().grid_y(),
    grid_z=contract.dimensions().grid_z(),
    block_x=contract.dimensions().block_x(),
    block_y=contract.dimensions().block_y(),
    block_z=contract.dimensions().block_z(),
    shared_memory_bytes=contract.dimensions().shared_memory_bytes(),
  ) catch {
    _ => raise InvalidPlan(LaunchDimensions)
  }
  let operands : Array[RegionBinding] = []
  for operand in contract.operands() {
    operands.push(bind_operand(operation.id(), operation_profile, operand))
  }
  if operation_profile.workspace().logical_bytes() == 0L &&
    contract.operands().any(operand => operand.role() is Workspace) {
    raise invalid_operation(operation.id(), Workspace)
  }
  {
    function_index: append_function_blueprint(
      artifacts,
      functions,
      contract.entry_point(),
    ),
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
  }
}

///|
fn vendor_step_blueprint(
  operation : @device_plan.StaticOperation,
  operation_profile : @device_profile.OperationExecutionProfile,
) -> VendorStepBlueprint raise DeviceExecutorError {
  if !(operation.kind() is (OutputProjection | LanguageModelHead)) {
    raise InvalidPlan(UnsupportedVendorOperation)
  }
  guard operation.shape() is ProjectionShape(input_width~, output_width~) else {
    raise invalid_operation(operation.id(), VendorShape)
  }
  guard operation_profile.inputs() is [Activation(left), Weight(_, weight)] &&
    operation_profile.outputs() is [output] else {
    raise invalid_operation(operation.id(), VendorInputs)
  }
  let rows = operation_profile.kernel_shape().token_rows()
  let inner = Int64::from_int(input_width)
  let columns = Int64::from_int(output_width)
  if left.region().logical_bytes() !=
    checked_bf16_bytes(rows, inner, operation.id()) ||
    weight.logical_bytes() != checked_bf16_bytes(columns, inner, operation.id()) ||
    output.region().logical_bytes() !=
    checked_bf16_bytes(rows, columns, operation.id()) {
    raise invalid_operation(operation.id(), VendorShape)
  }
  let workspace = operation_profile.workspace()
  if workspace.logical_bytes() != operation.binding().workspace_bytes() ||
    workspace.alignment_bytes() != operation.binding().workspace_alignment() ||
    (workspace.logical_bytes() > 0L && workspace.alignment_bytes() != 256L) {
    raise invalid_operation(operation.id(), Workspace)
  }
  {
    operation_id: operation.id(),
    rows,
    inner,
    columns,
    workspace_bytes: workspace.logical_bytes(),
    left_offset: left.region().offset_bytes(),
    weight_offset: weight.offset_bytes(),
    output_offset: output.region().offset_bytes(),
    workspace_offset: workspace.offset_bytes(),
  }
}

///|
fn build_blueprint_from_evidence(
  static_plan : @device_plan.StaticDevicePlan,
  memory_plan : @device_memory.DeviceMemoryPlan,
  profile : @device_profile.DeviceExecutionProfile,
  profile_id : @launch_contract.KernelProfileId,
  weight_layout : @device_materialize.DeviceWeightLayout,
  weight_allocation_bytes : Int64,
  contracts : @launch_contract.LaunchContractSet,
  artifacts : @artifact.KernelArtifactBundle,
  context_target_matches~ : Bool,
) -> PreparationBlueprint raise DeviceExecutorError {
  validate_top_level_evidence(
    static_plan, memory_plan, profile, weight_layout, weight_allocation_bytes, contracts,
    artifacts, context_target_matches,
  )
  ignore(selected_kernel_profile(contracts, profile, profile_id))
  guard static_plan.operations().get(0) is Some(first_operation) &&
    first_operation.kind() == Embedding &&
    first_operation.shape() is EmbeddingShape(vocabulary_size~, ..) else {
    raise InvalidPlan(OperationOrder)
  }
  let steps : Array[StepBlueprint] = []
  let functions : Array[FunctionBlueprint] = []
  let mut has_vendor_steps = false
  for index, operation in static_plan.operations() {
    if operation.id().as_int() != index {
      raise InvalidPlan(OperationOrder)
    }
    guard profile.operation(operation.id()) is Some(operation_profile) else {
      raise invalid_operation(operation.id(), ProfileMissing)
    }
    if operation_profile.operation_id() != operation.id() {
      raise invalid_operation(operation.id(), ProfileOrder)
    }
    match operation.binding().implementation() {
      ContentAddressedAot(_) => {
        guard contracts.contract(profile_id, operation.id()) is Some(contract) else {
          raise invalid_operation(operation.id(), ContractMissing)
        }
        steps.push(
          Aot(
            aot_step_blueprint(
              operation, operation_profile, contract, artifacts, functions,
            ),
          ),
        )
      }
      VendorBf16GemmImplicitHeuristicV1 => {
        has_vendor_steps = true
        steps.push(Vendor(vendor_step_blueprint(operation, operation_profile)))
      }
    }
  }
  if steps.length() != static_plan.operations().length() {
    raise InvalidPlan(OperationCount)
  }
  {
    token_staging_bytes: profile.token_staging().logical_bytes(),
    vocabulary_size,
    steps: ReadOnlyArray::from_array(steps),
    functions: ReadOnlyArray::from_array(functions),
    has_vendor_steps,
  }
}

///|
fn build_blueprint(
  context : @device.Context,
  static_plan : @device_plan.StaticDevicePlan,
  memory_plan : @device_memory.DeviceMemoryPlan,
  profile : @device_profile.DeviceExecutionProfile,
  profile_id : @launch_contract.KernelProfileId,
  weights : @device_materialize.DeviceWeights,
  contracts : @launch_contract.LaunchContractSet,
  artifacts : @artifact.KernelArtifactBundle,
) -> PreparationBlueprint raise DeviceExecutorError {
  build_blueprint_from_evidence(
    static_plan,
    memory_plan,
    profile,
    profile_id,
    weights.layout(),
    weights.allocation().byte_count(),
    contracts,
    artifacts,
    context_target_matches=static_plan
      .target()
      .matches_capability(context.capability()),
  )
}
