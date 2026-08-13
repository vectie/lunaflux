///|
fn empty_paged_resources() -> PagedGraphResources {
  {
    descriptor: None,
    stream: None,
    weight_lease: None,
    activation_workspace: None,
    persistent_kv: None,
    modules: [],
    functions: [],
  }
}

///|
fn executor_operation_error(
  operation : @plan.OperationId,
  rule : PagedGraphExecutorRule,
) -> DeviceStepError {
  InvalidExecutorOperation(operation, rule)
}

///|
fn validate_executor_evidence(
  context : @device.Context,
  blueprint : PagedKvExecutionBlueprint,
  weights : @device_materialize.DeviceWeights,
) -> Unit raise DeviceStepError {
  if blueprint.scope() != FullGraph {
    raise InvalidExecutor(BlueprintScope)
  }
  if !blueprint.target().matches_capability(context.capability()) {
    raise InvalidExecutor(ContextTarget)
  }
  if weights.identity().content() != blueprint.identity().content() {
    raise InvalidExecutor(WeightIdentity)
  }
  let weight_layout = weights.layout()
  if weight_layout.arena_bytes() != blueprint.weight_arena_bytes() ||
    weight_layout.alignment_bytes() != blueprint.weight_alignment() ||
    weights.allocation().byte_count() != blueprint.weight_arena_bytes() {
    raise InvalidExecutor(WeightLayout)
  }
  for step in blueprint.steps() {
    for operand in step.operands() {
      if operand.space() == WeightArena {
        guard operand.role() is WeightInput(reference) &&
          weights.region(reference) is Some(region) &&
          region.offset_bytes() == operand.offset_bytes() &&
          region.byte_length() == operand.byte_count() else {
          raise executor_operation_error(step.operation_id(), WeightRegion)
        }
      }
    }
  }
}

///|
fn descriptor_allocation(
  resources : DeviceStepResources,
  stage : DeviceStepStage,
) -> @device.Allocation {
  match stage {
    Counts => resources.counts
    TokenIds => resources.token_ids
    TokenPositions => resources.token_positions
    QueryRowOffsets => resources.query_row_offsets
    SequenceLengths => resources.sequence_lengths
    PageOffsets => resources.page_offsets
    PhysicalPageIndices => resources.physical_page_indices
  }
}

///|
fn allocation_for_paged_operand(
  binding : PagedKvOperandBinding,
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation_workspace : @device.Allocation,
  persistent_kv : @device.Allocation,
) -> @device.Allocation {
  match binding.space() {
    StepDescriptor(stage) => descriptor_allocation(descriptor, stage)
    WeightArena => weight
    ActivationWorkspaceArena => activation_workspace
    PersistentKvArena => persistent_kv
  }
}

///|
fn validate_paged_physical_regions(
  context : @device.Context,
  blueprint : PagedKvExecutionBlueprint,
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation_workspace : @device.Allocation,
  persistent_kv : @device.Allocation,
) -> Unit raise DeviceStepError {
  for step in blueprint.steps() {
    for binding in step.operands() {
      context.validate_allocation_region(
        allocation_for_paged_operand(
          binding, descriptor, weight, activation_workspace, persistent_kv,
        ),
        offset=binding.offset_bytes(),
        byte_count=binding.byte_count(),
        alignment=binding.alignment(),
      ) catch {
        error => raise ExecutorDevice(KernelArguments, error)
      }
    }
  }
}

///|
fn paged_function_index(
  functions : Array[PagedFunctionBlueprint],
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
fn collect_paged_functions(
  blueprint : PagedKvExecutionBlueprint,
) -> (ReadOnlyArray[PagedFunctionBlueprint], FixedArray[Int]) {
  let functions : Array[PagedFunctionBlueprint] = []
  let indices = FixedArray::make(blueprint.steps().length(), 0)
  for step_index, step in blueprint.steps() {
    match paged_function_index(functions, step.entry_point()) {
      Some(index) => indices[step_index] = index
      None => {
        functions.push({
          identity: step.entry_point(),
          module_index: step.artifact_module_index(),
          symbol: step.function_symbol(),
        })
        indices[step_index] = functions.length() - 1
      }
    }
  }
  (ReadOnlyArray::from_array(functions), indices)
}

///|
fn load_paged_modules(
  context : @device.Context,
  artifacts : @artifact.KernelArtifactBundle,
  resources : PagedGraphResources,
) -> Unit raise DeviceStepError {
  for image in artifacts.modules() {
    let resource = context.load_module(image.module_bytes()) catch {
      error => raise ExecutorDevice(Module, error)
    }
    resources.modules.push({ resource, open: true })
  }
}

///|
fn load_paged_functions(
  functions : ReadOnlyArray[PagedFunctionBlueprint],
  resources : PagedGraphResources,
) -> Unit raise DeviceStepError {
  for function in functions {
    guard resources.modules.get(function.module_index) is Some(owner) else {
      raise InvalidExecutor(ArtifactModule)
    }
    let resource = owner.resource.load_function(@utf8.encode(function.symbol)) catch {
      error => raise ExecutorDevice(Function, error)
    }
    resources.functions.push({
      resource,
      module_index: function.module_index,
      open: true,
    })
  }
}

///|
fn prepare_paged_steps(
  blueprint : PagedKvExecutionBlueprint,
  function_indices : FixedArray[Int],
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation_workspace : @device.Allocation,
  persistent_kv : @device.Allocation,
) -> ReadOnlyArray[PreparedPagedGraphStep] raise DeviceStepError {
  let prepared : Array[PreparedPagedGraphStep] = []
  for step_index, step in blueprint.steps() {
    let arguments : Array[@device.KernelArgument] = []
    for binding in step.operands() {
      arguments.push(
        @device.KernelArgument::device_region(
          allocation_for_paged_operand(
            binding, descriptor, weight, activation_workspace, persistent_kv,
          ),
          offset=binding.offset_bytes(),
          byte_count=binding.byte_count(),
          alignment=binding.alignment(),
        ) catch {
          error => raise ExecutorDevice(KernelArguments, error)
        },
      )
    }
    let packed = @device.KernelArguments::new(arguments) catch {
      error => raise ExecutorDevice(KernelArguments, error)
    }
    let source = step.dimensions()
    let dimensions = @device.KernelLaunchDimensions::new(
      grid_x=source.grid_x(),
      grid_y=source.grid_y(),
      grid_z=source.grid_z(),
      block_x=source.block_x(),
      block_y=source.block_y(),
      block_z=source.block_z(),
      shared_memory_bytes=source.shared_memory_bytes(),
    ) catch {
      _ => raise executor_operation_error(step.operation_id(), LaunchDimensions)
    }
    prepared.push({
      operation_id: step.operation_id(),
      function_index: function_indices[step_index],
      dimensions,
      arguments: packed,
    })
  }
  ReadOnlyArray::from_array(prepared)
}

///|
fn paged_construction_failed(
  resources : PagedGraphResources,
  cause : DeviceStepError,
) -> PagedGraphPreparation raise DeviceStepError {
  try close_paged_resources(resources) catch {
    cleanup_failure =>
      PagedCleanupRequired({
        resources,
        cause,
        cleanup_failure,
        lifecycle: GraphClosingFailed,
      })
  } noraise {
    _ => raise cause
  }
}

///|
/// Admits all immutable evidence before opening native resources, then owns
/// the complete synchronous full-graph paged execution resource set.
pub fn prepare_paged_graph_executor(
  context : @device.Context,
  model_plan : @plan.ModelPlan,
  model_generation : @worker_protocol.ModelPlanGeneration,
  layout : @device_layout.DeviceKvLayout,
  limits : DeviceStepLimits,
  static_plan : @device_plan.StaticDevicePlan,
  memory_plan : @device_memory.DeviceMemoryPlan,
  profile_id : @launch_contract.KernelProfileId,
  weights : @device_materialize.DeviceWeights,
  contracts : @launch_contract.PagedKvLaunchContractSet,
  artifacts : @artifact.KernelArtifactBundle,
  has_accepted_predecessor~ : Bool,
  accepted_predecessor_value~ : UInt64,
) -> PagedGraphPreparation raise DeviceStepError {
  let blueprint = admit_paged_graph_execution_blueprint(
    model_plan, limits, layout, static_plan, memory_plan, profile_id, contracts,
    artifacts,
  )
  validate_predecessor_seed(
    has_accepted_predecessor, accepted_predecessor_value,
  )
  validate_executor_evidence(context, blueprint, weights)
  let (functions, function_indices) = collect_paged_functions(blueprint)
  let resources = empty_paged_resources()
  let descriptor_outcome = prepare(
    context,
    model_plan,
    model_generation,
    layout,
    limits,
    has_accepted_predecessor~,
    accepted_predecessor_value~,
  ) catch {
    error => return paged_construction_failed(resources, error)
  }
  let descriptor = match descriptor_outcome {
    Ready(owner) => owner
    CleanupRequired(failed) => {
      resources.descriptor = Some(Failed(failed))
      return paged_construction_failed(resources, failed.cause())
    }
  }
  resources.descriptor = Some(Complete(descriptor))
  let weight = weights.allocation()
  let lease = context.lease_allocation(weight) catch {
    error =>
      return paged_construction_failed(
        resources,
        ExecutorDevice(WeightLease, error),
      )
  }
  resources.weight_lease = Some({ resource: lease, open: true })
  let stream = context.create_stream() catch {
    error =>
      return paged_construction_failed(resources, ExecutorDevice(Stream, error))
  }
  resources.stream = Some({ resource: stream, open: true })
  let activation_workspace = context.allocate(
    blueprint.activation_workspace_arena_bytes(),
  ) catch {
    error =>
      return paged_construction_failed(
        resources,
        ExecutorDevice(ActivationWorkspaceArena, error),
      )
  }
  resources.activation_workspace = Some({
    resource: activation_workspace,
    open: true,
  })
  let persistent_kv = context.allocate(layout.arena_bytes()) catch {
    error =>
      return paged_construction_failed(
        resources,
        ExecutorDevice(PersistentKvArena, error),
      )
  }
  resources.persistent_kv = Some({ resource: persistent_kv, open: true })
  if activation_workspace.byte_count() !=
    blueprint.activation_workspace_arena_bytes() ||
    persistent_kv.byte_count() != layout.arena_bytes() {
    return paged_construction_failed(resources, InvalidExecutor(AllocationSize))
  }
  validate_paged_physical_regions(
    context,
    blueprint,
    descriptor.resources,
    weight,
    activation_workspace,
    persistent_kv,
  ) catch {
    error => return paged_construction_failed(resources, error)
  }
  load_paged_modules(context, artifacts, resources) catch {
    error => return paged_construction_failed(resources, error)
  }
  load_paged_functions(functions, resources) catch {
    error => return paged_construction_failed(resources, error)
  }
  let steps = prepare_paged_steps(
    blueprint,
    function_indices,
    descriptor.resources,
    weight,
    activation_workspace,
    persistent_kv,
  ) catch {
    error => return paged_construction_failed(resources, error)
  }
  PagedReady({ resources, steps, lifecycle: GraphReady })
}
