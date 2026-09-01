///|
fn qwen_bucket_step_index(
  steps : ReadOnlyArray[PreparedPagedGraphStep],
  operation : @plan.OperationId,
) -> Int raise DeviceStepError {
  let mut found = -1
  for index, step in steps {
    if step.operation_id == operation {
      if found >= 0 {
        raise executor_operation_error(operation, GraphBucket)
      }
      found = index
    }
  }
  if found < 0 {
    raise executor_operation_error(operation, GraphBucket)
  }
  found
}

///|
/// Derives the only currently safe smaller-bucket launch contract from the
/// approved Qwen composite ABI. Postprocess uses one block per query token while
/// readonly attention uses a bounded grid-stride token traversal.
fn prepare_qwen_graph_launch_contract(
  blueprint : PagedKvExecutionBlueprint,
  steps : ReadOnlyArray[PreparedPagedGraphStep],
  spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
) -> PagedGraphLaunchContract? raise DeviceStepError {
  if spans.is_empty() {
    return None
  }
  let launches = FixedArray::make(steps.length(), FixedBucketLaunch)
  for span in spans {
    let (_, qk_norm_index, _, attention_index) = span.operation_indices()
    guard blueprint.steps().get(qk_norm_index) is Some(qk_norm) &&
      blueprint.steps().get(attention_index) is Some(attention) else {
      raise InvalidExecutor(GraphBucket)
    }
    let postprocess = span.postprocess_launch()
    let readonly_launch = span.readonly_launch()
    if postprocess.grid_x() !=
      Int::min(blueprint.profile().max_query_tokens(), 4096) ||
      readonly_launch.grid_x() <= 0 ||
      readonly_launch.grid_x() > blueprint.profile().max_query_tokens() {
      raise InvalidExecutor(GraphBucket)
    }
    let postprocess_index = qwen_bucket_step_index(
      steps,
      qk_norm.operation_id(),
    )
    let readonly_index = qwen_bucket_step_index(steps, attention.operation_id())
    match (launches[postprocess_index], launches[readonly_index]) {
      (FixedBucketLaunch, FixedBucketLaunch) => {
        launches[postprocess_index] = QueryTokenCappedGridX(postprocess)
        launches[readonly_index] = QueryTokenCappedGridX(readonly_launch)
      }
      _ => raise InvalidExecutor(GraphBucket)
    }
  }
  Some({ step_launches: launches })
}

///|
fn paged_scaled_query_token_launch(
  source : @launch_contract.AotLaunchDimensions,
  query_token_bound : Int,
) -> @launch_contract.AotLaunchDimensions raise DeviceStepError {
  if query_token_bound <= 0 || query_token_bound > source.grid_x() {
    raise InvalidExecutor(GraphBucket)
  }
  @launch_contract.AotLaunchDimensions::new(
    grid_x=query_token_bound,
    grid_y=source.grid_y(),
    grid_z=source.grid_z(),
    block_x=source.block_x(),
    block_y=source.block_y(),
    block_z=source.block_z(),
    shared_memory_bytes=source.shared_memory_bytes(),
  ) catch {
    _ => raise InvalidExecutor(GraphBucket)
  }
}

///|
fn paged_capped_query_token_launch(
  source : @launch_contract.AotLaunchDimensions,
  query_token_bound : Int,
) -> @launch_contract.AotLaunchDimensions raise DeviceStepError {
  if query_token_bound <= 0 {
    raise InvalidExecutor(GraphBucket)
  }
  paged_scaled_query_token_launch(
    source,
    Int::min(source.grid_x(), query_token_bound),
  )
}

///|
fn paged_device_launch_dimensions(
  source : @launch_contract.AotLaunchDimensions,
) -> @device.KernelLaunchDimensions raise DeviceStepError {
  @device.KernelLaunchDimensions::new(
    grid_x=source.grid_x(),
    grid_y=source.grid_y(),
    grid_z=source.grid_z(),
    block_x=source.block_x(),
    block_y=source.block_y(),
    block_z=source.block_z(),
    shared_memory_bytes=source.shared_memory_bytes(),
  ) catch {
    _ => raise InvalidExecutor(GraphBucket)
  }
}

///|
fn paged_ordered_launches(
  steps : ReadOnlyArray[PreparedPagedGraphStep],
  resources : PagedGraphResources,
  query_token_bound? : Int? = None,
) -> Array[@device.OrderedKernelLaunch] raise DeviceStepError {
  let launches : Array[@device.OrderedKernelLaunch] = []
  for index, step in steps {
    guard resources.functions.get(step.function_index) is Some(function) &&
      function.open else {
      raise InvalidExecutor(ResourceState)
    }
    let dimensions = match
      (query_token_bound, resources.graph_launch_contract) {
      (Some(bound), Some(contract)) =>
        match contract.step_launches.get(index) {
          Some(QueryTokenCappedGridX(source)) =>
            paged_device_launch_dimensions(
              paged_capped_query_token_launch(source, bound),
            )
          _ => step.dimensions
        }
      _ => step.dimensions
    }
    launches.push(
      @device.OrderedKernelLaunch::new(
        function.resource,
        dimensions,
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
  operation_ids : ReadOnlyArray[@plan.OperationId],
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
  if operation_ids.length() != launches.length() {
    raise InvalidExecutor(ResourceState)
  }
  resources.ordered_executors.push({
    resource: executor,
    operation_ids,
    open: true,
  })
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
  decode_steps? : ReadOnlyArray[PreparedPagedGraphStep]? = None,
) -> PagedOrderedExecutorDispatch raise DeviceStepError {
  let maximum_launches = paged_ordered_launches(steps, resources)
  let baseline_operation_ids = ReadOnlyArray::from_array(
    steps.map(step => step.operation_id),
  )
  let slot_owner_indices = FixedArray::make(table.bucket_count(), -1)
  let mut fallback_owner_index = -1
  let telemetry_owner_index = match graph_policy {
    PagedOrderedEagerOnly => {
      fallback_owner_index = prepare_one_paged_ordered_executor(
        context,
        maximum_launches[:],
        baseline_operation_ids,
        resources,
        OrderedKernelEagerOnly,
      )
      fallback_owner_index
    }
    PagedCapturedRequired | PagedCapturedWithEagerFallback => {
      if resources.graph_launch_contract is Some(_) {
        for bound in paged_decode_capture_bounds(profile) {
          let launches = paged_ordered_launches(
            steps,
            resources,
            query_token_bound=Some(bound),
          )
          let owner = prepare_one_paged_ordered_executor(
            context,
            launches[:],
            baseline_operation_ids,
            resources,
            paged_ordered_device_policy(graph_policy),
          )
          map_paged_decode_capture_owner(
            table, bound, bound, owner, slot_owner_indices,
          )
        }
      }
      let maximum = prepare_one_paged_ordered_executor(
        context,
        maximum_launches[:],
        baseline_operation_ids,
        resources,
        paged_ordered_device_policy(graph_policy),
      )
      for slot in paged_maximum_graph_slots(table, profile) {
        if slot_owner_indices[slot] < 0 {
          slot_owner_indices[slot] = maximum
        }
      }
      fallback_owner_index = match graph_policy {
        PagedCapturedRequired => maximum
        PagedCapturedWithEagerFallback =>
          prepare_one_paged_ordered_executor(
            context,
            maximum_launches[:],
            baseline_operation_ids,
            resources,
            OrderedKernelEagerOnly,
          )
        PagedOrderedEagerOnly => abort("unreachable graph policy")
      }
      maximum
    }
  }
  let decode_split_owner_index = match decode_steps {
    Some(selected) => {
      let launches = paged_ordered_launches(selected, resources)
      let operation_ids = ReadOnlyArray::from_array(
        selected.map(step => step.operation_id),
      )
      prepare_one_paged_ordered_executor(
        context,
        launches[:],
        operation_ids,
        resources,
        paged_ordered_device_policy(graph_policy),
      )
    }
    None => -1
  }
  {
    slot_owner_indices,
    fallback_owner_index,
    telemetry_owner_index,
    decode_split_owner_index,
  }
}
