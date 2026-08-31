///|
fn empty_paged_resources() -> PagedGraphResources {
  {
    descriptor: None,
    stream: None,
    weight_lease: None,
    activation_workspace: None,
    persistent_kv: None,
    fused_dispatch_canary: None,
    greedy_sampling_results: None,
    modules: [],
    functions: [],
    ordered_executor: None,
    numeric_weights: None,
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
fn paged_function_symbol_index(
  functions : Array[PagedFunctionBlueprint],
  module_index : Int,
  symbol : StringView,
) -> Int? {
  for index, function in functions {
    if function.module_index == module_index &&
      function.symbol == symbol.to_owned() {
      return Some(index)
    }
  }
  None
}

///|
fn paged_step_has_split_decode_mlp(step : PagedKvExecutionStep) -> Bool {
  step.operation_kind() == GatedMlp &&
  step.operands().any(operand => operand.role() == Workspace)
}

///|
fn collect_paged_functions(
  blueprint : PagedKvExecutionBlueprint,
) -> (ReadOnlyArray[PagedFunctionBlueprint], FixedArray[Int], FixedArray[Int]) {
  let functions : Array[PagedFunctionBlueprint] = []
  let indices = FixedArray::make(blueprint.steps().length(), 0)
  let split_decode_indices = FixedArray::make(blueprint.steps().length(), -1)
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
  for step_index, step in blueprint.steps() {
    if paged_step_has_split_decode_mlp(step) {
      let symbol = step.function_symbol() + "_down"
      match
        paged_function_symbol_index(
          functions,
          step.artifact_module_index(),
          symbol,
        ) {
        Some(index) => split_decode_indices[step_index] = index
        None => {
          functions.push({
            identity: step.entry_point(),
            module_index: step.artifact_module_index(),
            symbol,
          })
          split_decode_indices[step_index] = functions.length() - 1
        }
      }
    }
  }
  (ReadOnlyArray::from_array(functions), indices, split_decode_indices)
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
fn prepare_paged_ordered_executor(
  context : @device.Context,
  steps : ReadOnlyArray[PreparedPagedGraphStep],
  resources : PagedGraphResources,
  graph_policy : PagedGraphExecutionPolicy,
) -> Unit raise DeviceStepError {
  guard resources.stream is Some(stream) && stream.open else {
    raise InvalidExecutor(ResourceState)
  }
  let launches : Array[@device.OrderedKernelLaunch] = []
  for step in steps {
    guard resources.functions.get(step.function_index) is Some(function) &&
      function.open else {
      raise InvalidExecutor(ResourceState)
    }
    launches.push(
      @device.OrderedKernelLaunch::new(
        function.resource,
        step.dimensions,
        step.arguments,
      ),
    )
  }
  let executor = context.create_ordered_kernel_executor(
    stream.resource,
    launches[:],
    policy=paged_ordered_device_policy(graph_policy),
  ) catch {
    error => raise ExecutorDevice(OrderedExecutor, error)
  }
  resources.ordered_executor = Some({ resource: executor, open: true })
}

///|
fn paged_ordered_device_policy(
  policy : PagedGraphExecutionPolicy,
) -> @device.OrderedKernelExecutionPolicy {
  match policy {
    PagedOrderedEagerOnly => OrderedKernelEagerOnly
    PagedCapturedRequired => OrderedKernelCaptureRequired
    PagedCapturedWithEagerFallback => OrderedKernelCaptureWithEagerFallback
  }
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
  luna_authorization? : @luna_artifact_admission.LunaAotKernelAdmission? = None,
  graph_policy? : PagedGraphExecutionPolicy = PagedOrderedEagerOnly,
  fused_residual_spans? : ReadOnlyArray[ApprovedFusedResidualRmsNormSpan] = ReadOnlyArray::from_array([],
  ),
  fused_qkv_readonly_span? : ApprovedFusedQkvReadOnlyAttentionSpan? = None,
  diagnostic_policy? : PagedExecutionDiagnosticPolicy = ProductionFastPath,
  sampling_policy? : PagedSamplingExecutionPolicy = PagedHostSampling,
) -> PagedGraphPreparation raise DeviceStepError {
  let blueprint = admit_paged_graph_execution_blueprint(
    model_plan,
    limits,
    layout,
    static_plan,
    memory_plan,
    profile_id,
    contracts,
    artifacts,
    luna_authorization~,
  )
  validate_predecessor_seed(
    has_accepted_predecessor, accepted_predecessor_value,
  )
  validate_executor_evidence(context, blueprint, weights)
  for span in fused_residual_spans {
    if span.model_generation() != model_generation ||
      span.target() != blueprint.target() ||
      !fused_dispatch_policy_allows_graph(diagnostic_policy, graph_policy) ||
      !fused_dispatch_policy_allows_span(diagnostic_policy, span.kernel_abi()) {
      raise InvalidExecutor(ResourceState)
    }
  }
  match fused_qkv_readonly_span {
    Some(span) =>
      if span.model_generation() != model_generation ||
        span.target() != blueprint.target() {
        raise InvalidExecutor(ResourceState)
      }
    None => ()
  }
  let completion = prepare_paged_completion_scratch(
    model_plan, memory_plan, limits,
  )
  let (functions, function_indices, split_decode_indices) = collect_paged_functions(
    blueprint,
  )
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
  let model_steps = prepare_paged_production_steps(
    context,
    blueprint,
    function_indices,
    split_decode_indices,
    descriptor.resources,
    weight,
    activation_workspace,
    persistent_kv,
    resources,
    fused_residual_spans,
    fused_qkv_readonly_span,
  ) catch {
    error => return paged_construction_failed(resources, error)
  }
  finish_paged_sampling_preparation(
    context,
    model_plan,
    blueprint,
    artifacts,
    limits,
    sampling_policy,
    completion,
    descriptor.resources,
    activation_workspace,
    model_steps,
    resources,
    graph_policy,
    fused_residual_spans,
    diagnostic_policy,
  )
}
