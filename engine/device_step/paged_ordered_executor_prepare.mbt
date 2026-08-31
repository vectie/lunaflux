///|
fn paged_ordered_launches(
  steps : ReadOnlyArray[PreparedPagedGraphStep],
  resources : PagedGraphResources,
) -> Array[@device.OrderedKernelLaunch] raise DeviceStepError {
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
  launches
}

///|
fn prepare_one_paged_ordered_executor(
  context : @device.Context,
  launches : ArrayView[@device.OrderedKernelLaunch],
  resources : PagedGraphResources,
  policy : @device.OrderedKernelExecutionPolicy,
) -> Int raise DeviceStepError {
  guard resources.stream is Some(stream) && stream.open else {
    raise InvalidExecutor(ResourceState)
  }
  let executor = context.create_ordered_kernel_executor(
    stream.resource,
    launches,
    policy~,
  ) catch {
    error => raise ExecutorDevice(OrderedExecutor, error)
  }
  resources.ordered_executors.push({ resource: executor, open: true })
  resources.ordered_executors.length() - 1
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
fn prepare_paged_ordered_executors(
  context : @device.Context,
  steps : ReadOnlyArray[PreparedPagedGraphStep],
  resources : PagedGraphResources,
  graph_policy : PagedGraphExecutionPolicy,
  table : @luna_execution_graph_strategy.ExecutionGraphBucketTable,
  profile : @launch_contract.PagedKernelProfile,
) -> PagedOrderedExecutorDispatch raise DeviceStepError {
  let launches = paged_ordered_launches(steps, resources)
  let exact_slots = paged_exact_maximum_graph_slots(table, profile)
  let exact_slot_owner_indices = FixedArray::make(3, -1)
  let mut fallback_owner_index = -1
  let telemetry_owner_index = match graph_policy {
    PagedOrderedEagerOnly => {
      fallback_owner_index = prepare_one_paged_ordered_executor(
        context,
        launches[:],
        resources,
        paged_ordered_device_policy(graph_policy),
      )
      fallback_owner_index
    }
    PagedCapturedRequired => {
      let exact = prepare_one_paged_ordered_executor(
        context,
        launches[:],
        resources,
        paged_ordered_device_policy(graph_policy),
      )
      for index in 0..<3 {
        exact_slot_owner_indices[index] = exact
      }
      // The admitted launch list is still one max-envelope graph. Required
      // capture therefore preserves its existing legal row-guard fallback for
      // smaller live counts instead of rejecting production traffic.
      fallback_owner_index = exact
      exact
    }
    PagedCapturedWithEagerFallback => {
      let exact = prepare_one_paged_ordered_executor(
        context,
        launches[:],
        resources,
        paged_ordered_device_policy(graph_policy),
      )
      for index in 0..<3 {
        exact_slot_owner_indices[index] = exact
      }
      fallback_owner_index = prepare_one_paged_ordered_executor(
        context,
        launches[:],
        resources,
        OrderedKernelEagerOnly,
      )
      exact
    }
  }
  {
    exact_slots,
    exact_slot_owner_indices,
    fallback_owner_index,
    telemetry_owner_index,
  }
}
