///|
fn empty_worker_resources() -> TensorParallelPartialResources {
  {
    collective: None,
    executor: None,
    functions: [],
    modules: [],
    stream: None,
    kv: None,
    activation_workspace: None,
    rank_constants: None,
    descriptor: None,
    weights: None,
    context: None,
  }
}

///|
fn require_context(
  resources : TensorParallelPartialResources,
) -> @device.Context raise TensorParallelDeviceWorkerError {
  guard resources.context is Some(context) else { raise Invalid(ResourceState) }
  context
}

///|
fn require_weights(
  resources : TensorParallelPartialResources,
) -> @device_materialize.DeviceShardedWeights raise TensorParallelDeviceWorkerError {
  guard resources.weights is Some(WeightComplete(weights)) else {
    raise Invalid(ResourceState)
  }
  weights
}

///|
fn require_descriptor(
  resources : TensorParallelPartialResources,
) -> @device_step.DeviceStepOwner raise TensorParallelDeviceWorkerError {
  guard resources.descriptor is Some(DescriptorComplete(owner)) else {
    raise Invalid(ResourceState)
  }
  owner
}

///|
fn require_kv(
  resources : TensorParallelPartialResources,
) -> @device_arena.DeviceKvArena raise TensorParallelDeviceWorkerError {
  guard resources.kv is Some(KvComplete(owner)) else {
    raise Invalid(ResourceState)
  }
  owner
}

///|
fn require_allocation(
  resource : TensorParallelOwnedAllocation?,
) -> @device.Allocation raise TensorParallelDeviceWorkerError {
  guard resource is Some(owner) && owner.open else {
    raise Invalid(ResourceState)
  }
  owner.resource
}

///|
fn take_live_resources(
  partial : TensorParallelPartialResources,
) -> TensorParallelLiveResources raise TensorParallelDeviceWorkerError {
  guard partial.collective is Some(collective) && collective.open else {
    raise Invalid(ResourceState)
  }
  guard partial.executor is Some(executor) && executor.open else {
    raise Invalid(ResourceState)
  }
  guard partial.stream is Some(stream) && stream.open else {
    raise Invalid(ResourceState)
  }
  guard partial.kv is Some(KvComplete(kv)) else { raise Invalid(ResourceState) }
  guard partial.activation_workspace is Some(activation_workspace) &&
    activation_workspace.open else {
    raise Invalid(ResourceState)
  }
  guard partial.rank_constants is Some(rank_constants) && rank_constants.open else {
    raise Invalid(ResourceState)
  }
  guard partial.descriptor is Some(DescriptorComplete(descriptor)) else {
    raise Invalid(ResourceState)
  }
  guard partial.weights is Some(WeightComplete(weights)) else {
    raise Invalid(ResourceState)
  }
  guard partial.context is Some(context) else { raise Invalid(ResourceState) }
  let functions = partial.functions
  let modules = partial.modules
  partial.collective = None
  partial.executor = None
  partial.functions = []
  partial.modules = []
  partial.stream = None
  partial.kv = None
  partial.activation_workspace = None
  partial.rank_constants = None
  partial.descriptor = None
  partial.weights = None
  partial.context = None
  {
    collective,
    executor,
    functions,
    modules,
    stream,
    kv,
    activation_workspace,
    rank_constants,
    descriptor,
    weights,
    context,
  }
}

///|
fn allocate_rank_constants(
  context : @device.Context,
  plan : TensorParallelDeviceWorkerPlan,
  resources : TensorParallelPartialResources,
) -> Unit raise TensorParallelDeviceWorkerError {
  let region = plan.execution.rank_constants()
  if region.offset_bytes() != 0L ||
    region.byte_count() <= 0L ||
    region.byte_count() > 4096L {
    raise Invalid(RankConstants)
  }
  let allocation = context.allocate(region.reserved_bytes()) catch {
    error => raise Device(RankConstants, error)
  }
  resources.rank_constants = Some({ resource: allocation, open: true })
  if allocation.byte_count() != region.reserved_bytes() {
    raise Invalid(RankConstants)
  }
  context.validate_allocation_region(
    allocation,
    offset=0L,
    byte_count=region.byte_count(),
    alignment=region.alignment(),
  ) catch {
    error => raise Device(RankConstants, error)
  }
  let host = FixedArray::make(region.byte_count().to_int(), b'\x00')
  plan.rank_constants.encode_into(host, destination_offset=0) catch {
    _ => raise Invalid(RankConstants)
  }
  context.copy_from_fixed_host(allocation, host, 0L, 0L, region.byte_count()) catch {
    error => raise Device(RankConstants, error)
  }
}

///|
fn allocate_activation_workspace(
  context : @device.Context,
  plan : TensorParallelDeviceWorkerPlan,
  resources : TensorParallelPartialResources,
) -> Unit raise TensorParallelDeviceWorkerError {
  let bytes = plan.execution.activation_workspace_bytes()
  if bytes <= 0L {
    raise Invalid(ActivationWorkspace)
  }
  let allocation = context.allocate(bytes) catch {
    error => raise Device(ActivationWorkspace, error)
  }
  resources.activation_workspace = Some({ resource: allocation, open: true })
  if allocation.byte_count() != bytes {
    raise Invalid(ActivationWorkspace)
  }
}

///|
fn prepare_completion_authority(
  plan : TensorParallelDeviceWorkerPlan,
) -> TensorParallelCompletionAuthority raise TensorParallelDeviceWorkerError {
  if plan.execution.rank() != 0 {
    return FollowerAuthority
  }
  let vocabulary_size = plan.model.shape_constraints().vocabulary_size()
  let row_bytes = checked_product(vocabulary_size.to_int64(), 2L)
  guard plan.execution.activation(plan.model.final_output()) is Some(output) else {
    raise Invalid(FinalOutput)
  }
  let completion = @worker_wire.CompletionFrameBuffer::new(
    plan.startup.worker_limits(),
    plan.startup.inference_limits(),
  ) catch {
    error => raise Wire(error)
  }
  let sampling_limits = @sampling.SamplingLimits::new(
    plan.startup.worker_limits().max_plan_rows(),
    vocabulary_size,
  ) catch {
    error => raise Sampling(error)
  }
  let sampler = @sampling.SamplerScratch::new(vocabulary_size, sampling_limits) catch {
    error => raise Sampling(error)
  }
  LeaderAuthority({
    completion,
    sampling_limits,
    sampler,
    encoded_logits: FixedArray::make(row_bytes.to_int(), b'\x00'),
    logits: FixedArray::make(vocabulary_size, 0.0),
    selected_tokens: FixedArray::make(
      plan.startup.worker_limits().max_plan_rows(),
      0,
    ),
    final_output_offset: output.region().offset_bytes(),
    row_bytes,
    vocabulary_size,
  })
}

///|
/// Opens and binds every rank-local resource only after inert admission has
/// rebuilt and authenticated the exact execution plan. Any failure after the
/// first resource opens either cleans everything or returns retry authority.
pub fn prepare(
  model_root : @approved_fs.ApprovedRoot,
  collective_runtime : @device.TensorCollectiveRuntime,
  plan : TensorParallelDeviceWorkerPlan,
) -> TensorParallelDeviceWorkerPreparation raise TensorParallelDeviceWorkerError {
  if collective_runtime.collective_runtime_version_code() !=
    plan.collective_runtime_version_code {
    raise Invalid(CollectiveOrder)
  }
  let resources = empty_worker_resources()
  let context = @device.open_context(plan.execution.device_ordinal()) catch {
    error => raise Device(Context, error)
  }
  resources.context = Some(context)
  if context.ordinal() != plan.execution.device_ordinal() {
    return preparation_failed(resources, Invalid(DeviceOrdinal))
  }
  if !plan.execution.target().matches_capability(context.capability()) {
    return preparation_failed(resources, Invalid(DeviceTarget))
  }
  let weight_outcome = @device_materialize.load_inspected_sharded_file(
    context,
    model_root,
    plan.inspection,
  ) catch {
    error => return preparation_failed(resources, Weights(error))
  }
  match weight_outcome {
    ShardedWeightsReady(weights) => {
      resources.weights = Some(WeightComplete(weights))
      if weights.identity() != plan.model.identity() ||
        weights.rank_plan() != plan.inspection.rank_plan() ||
        weights.allocation().byte_count() != plan.execution.weight_arena_bytes() {
        return preparation_failed(resources, Invalid(WeightBinding))
      }
    }
    ShardedWeightsCleanupRequired(failed) => {
      resources.weights = Some(WeightFailed(failed))
      return preparation_failed(resources, Weights(failed.cause()))
    }
  }
  let descriptor_outcome = @device_step.prepare_rank_local(
    context,
    plan.model,
    plan.model_generation,
    plan.kv_plan,
    plan.execution.rank(),
    plan.descriptor_limits,
    has_accepted_predecessor=plan.startup.has_predecessor(),
    accepted_predecessor_value=plan.startup.predecessor_value(),
  ) catch {
    error => return preparation_failed(resources, Descriptor(error))
  }
  match descriptor_outcome {
    Ready(owner) => resources.descriptor = Some(DescriptorComplete(owner))
    CleanupRequired(failed) => {
      resources.descriptor = Some(DescriptorFailed(failed))
      return preparation_failed(resources, Descriptor(failed.cause()))
    }
  }
  allocate_rank_constants(context, plan, resources) catch {
    error => return preparation_failed(resources, error)
  }
  allocate_activation_workspace(context, plan, resources) catch {
    error => return preparation_failed(resources, error)
  }
  let layout = plan.kv_plan.rank_layout(plan.execution.rank()) catch {
    _ => return preparation_failed(resources, Invalid(KvBinding))
  }
  let kv_outcome = @device_arena.prepare(context, plan.model.identity(), layout) catch {
    error => return preparation_failed(resources, Kv(error))
  }
  match kv_outcome {
    Ready(owner) => resources.kv = Some(KvComplete(owner))
    CleanupRequired(failed) => {
      resources.kv = Some(KvFailed(failed))
      return preparation_failed(resources, Kv(failed.cause()))
    }
  }
  let stream = context.create_stream() catch {
    error => return preparation_failed(resources, Device(Stream, error))
  }
  resources.stream = Some({ resource: stream, open: true })
  let prepared = prepare_physical_steps(plan, resources) catch {
    error => return preparation_failed(resources, error)
  }
  let launches = prepared.launches
  let executor = context.create_ordered_kernel_executor(stream, launches[:]) catch {
    error => return preparation_failed(resources, Device(KernelExecutor, error))
  }
  resources.executor = Some({
    resource: executor,
    open: true,
    abort_required: false,
  })
  let completion = prepare_completion_authority(plan) catch {
    error => return preparation_failed(resources, error)
  }
  let collective = context.create_tensor_collective_owner(
    runtime=collective_runtime,
    group_id=plan.rendezvous_id,
    contract=plan.collective_contract,
    generation=plan.group_generation.value(),
    previous_plan_sequence_value=plan.startup.predecessor_value(),
  ) catch {
    error => return preparation_failed(resources, Device(Collective, error))
  }
  resources.collective = Some({
    resource: collective,
    open: true,
    abort_required: true,
  })
  let live_resources = take_live_resources(resources) catch {
    error => return preparation_failed(resources, error)
  }
  TensorParallelWorkerPrepared({
    plan,
    resources: live_resources,
    steps: prepared.steps,
    completion,
    lifecycle: WorkerStarting,
    active_epoch: 0UL,
    active_sequence: 0UL,
    active_rows: 0,
    active_tokens: 0,
    operation_index: 0,
    collective_index: 0,
    next_global_collective_sequence: 1UL,
    graph_hits: 0UL,
    graph_misses: 0UL,
    last_graph_sequence: 0UL,
    primary_failure: None,
    failure_class: None,
    cleanup_failure: None,
  })
}
