///|
/// Ordinary AOT execution has no asserted specialization byte bound. Limit
/// its optional variants structurally. Existing bounded specializations keep
/// their exact capture set until their declared byte budget includes variants.
fn effect_capture_budget(
  accounting : PagedGraphMemoryAvailability,
) -> @luna_execution_graph_strategy.ExecutionGraphVariantBudget {
  let enabled = accounting == GraphMemoryAbsent
  try! @luna_execution_graph_strategy.ExecutionGraphVariantBudget::new(
    if enabled {
      64
    } else {
      0
    },
    if enabled {
      32768
    } else {
      0
    },
  )
}

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
/// Share existing buffers/functions and specialize an immutable launch list
/// for effects only. Optional captures have a startup graph/node budget; this
/// is not a claim about CUDA's private graph-storage byte usage.
fn prepare_paged_effect_only_executor(
  context : @device.Context,
  launches : ArrayView[@device.OrderedKernelLaunch],
  operation_ids : ReadOnlyArray[@plan.OperationId],
  resources : PagedGraphResources,
  owner_index : Int,
) -> Unit raise DeviceStepError {
  let owner = resources.ordered_executors[owner_index]
  let count = effect_only_launch_count(
    operation_ids,
    resources.effect_operation_prefix,
  )
  if count >= launches.length() {
    return
  }
  let captured = owner.resource.execution_mode() == OrderedKernelCaptured
  let reservation = if captured {
    resources.effect_capture_budget.reserve(count)
  } else {
    None
  }
  // Exhaustion preserves the existing capture and its arithmetic predicate.
  // Do not replace a full capture with a many-launch eager prefix.
  if captured && reservation is None {
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
    policy=if captured {
      OrderedKernelCaptureWithEagerFallback
    } else {
      OrderedKernelEagerOnly
    },
  ) catch {
    error => {
      // Optional resource pressure leaves the already-created full executor
      // available. The device constructor owns rollback of a failed create.
      if captured {
        return
      }
      raise ExecutorDevice(OrderedExecutor, error)
    }
  }
  let prefix_index = resources.ordered_executors.length()
  resources.ordered_executors.push({
    resource: executor,
    operation_ids: prefix_ids,
    effect_only_owner_index: -1,
    mixed_owner_index: -1,
    open: true,
  })
  if captured && executor.execution_mode() != OrderedKernelCaptured {
    // Track ownership before closing so failed release is retryable by the
    // normal construction cleanup. Never dispatch this slower eager fallback.
    executor.close() catch {
      error => raise ExecutorDevice(OrderedExecutor, error)
    }
    resources.ordered_executors[prefix_index].open = false
    return
  }
  owner.effect_only_owner_index = prefix_index
  if reservation is Some(remaining) {
    resources.effect_capture_budget = remaining
  }
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
