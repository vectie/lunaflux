///|
/// Translate a semantic prefix to a lowered launch prefix. Repeated operation
/// IDs cover split kernels. Fusions may conservatively retain extra pure work.
/// Unknown/reordered mappings keep the full executor rather than omit effects.
fn effect_only_launch_count(
  operations : ReadOnlyArray[@plan.OperationId],
  semantic_prefix : Int,
) -> Int {
  if semantic_prefix <= 0 {
    return operations.length()
  }
  let mut count = 0
  let mut previous = -1
  for index, operation in operations {
    let value = operation.as_int()
    if value < previous {
      return operations.length()
    }
    if value < semantic_prefix {
      count = index + 1
    }
    previous = value
  }
  if count == 0 {
    operations.length()
  } else {
    count
  }
}

///|
/// Eager queues can share all existing buffers/functions and retain a shorter
/// immutable launch list. Captures are deliberately not duplicated here: their
/// extra storage must first be included in the graph-memory budget.
fn prepare_paged_effect_only_executor(
  context : @device.Context,
  launches : ArrayView[@device.OrderedKernelLaunch],
  operation_ids : ReadOnlyArray[@plan.OperationId],
  resources : PagedGraphResources,
  owner_index : Int,
) -> Unit raise DeviceStepError {
  let owner = resources.ordered_executors[owner_index]
  if owner.resource.execution_mode() != OrderedKernelEager {
    return
  }
  let count = effect_only_launch_count(
    operation_ids,
    resources.effect_operation_prefix,
  )
  if count >= launches.length() {
    return
  }
  guard resources.stream is Some(stream) && stream.open else {
    raise InvalidExecutor(ResourceState)
  }
  let prefix_ids = ReadOnlyArray::from_array(
    Array::makei(count, index => operation_ids[index]),
  )
  let executor = context.create_ordered_kernel_executor(
    stream.resource,
    launches[:count],
    policy=OrderedKernelEagerOnly,
  ) catch {
    error => raise ExecutorDevice(OrderedExecutor, error)
  }
  let prefix_index = resources.ordered_executors.length()
  resources.ordered_executors.push({
    resource: executor,
    operation_ids: prefix_ids,
    effect_only_owner_index: -1,
    open: true,
  })
  owner.effect_only_owner_index = prefix_index
}

///|
fn select_paged_output_demand(
  executor : PagedGraphExecutor,
  owner_index : Int,
  output_rows : Int,
) -> Int {
  // Diagnostic canaries and FP8 envelopes observe a full graph explicitly.
  if output_rows != 0 ||
    executor.fused_dispatch_canary is Some(_) ||
    executor.fp8_frame_envelope is Some(_) {
    return owner_index
  }
  let prefix = executor.resources.ordered_executors[owner_index].effect_only_owner_index
  if prefix >= 0 {
    prefix
  } else {
    owner_index
  }
}
