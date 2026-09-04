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
fn qwen_bucket_step_indices(
  steps : ReadOnlyArray[PreparedPagedGraphStep],
  operation : @plan.OperationId,
) -> ReadOnlyArray[Int] {
  let found : Array[Int] = []
  for index, step in steps {
    if step.operation_id == operation {
      found.push(index)
    }
  }
  ReadOnlyArray::from_array(found)
}

///|
fn paged_projection_output_width(
  blueprint : PagedKvExecutionBlueprint,
  step : PagedKvExecutionStep,
) -> Int raise DeviceStepError {
  let mut bytes = -1L
  for operand in step.operands() {
    if operand.role() is ActivationOutput(_) {
      if bytes >= 0L {
        raise executor_operation_error(step.operation_id(), GraphBucket)
      }
      bytes = operand.byte_count()
    }
  }
  let row_bytes = blueprint.profile().max_query_tokens().to_int64() * 2L
  if bytes <= 0L || row_bytes <= 0L || bytes % row_bytes != 0L {
    raise executor_operation_error(step.operation_id(), GraphBucket)
  }
  let width = bytes / row_bytes
  if width <= 0L || width > 2147483647L {
    raise executor_operation_error(step.operation_id(), GraphBucket)
  }
  width.to_int()
}

///|
fn paged_projection_intermediate_width(
  blueprint : PagedKvExecutionBlueprint,
  step : PagedKvExecutionStep,
) -> Int raise DeviceStepError {
  let mut bytes = -1L
  for operand in step.operands() {
    if operand.role() == Workspace {
      if bytes >= 0L {
        raise executor_operation_error(step.operation_id(), GraphBucket)
      }
      bytes = operand.byte_count()
    }
  }
  let row_bytes = blueprint.profile().max_query_tokens().to_int64() * 2L
  if bytes <= 0L || row_bytes <= 0L || bytes % row_bytes != 0L {
    raise executor_operation_error(step.operation_id(), GraphBucket)
  }
  let width = bytes / row_bytes
  if width <= 0L || width > 2147483647L {
    raise executor_operation_error(step.operation_id(), GraphBucket)
  }
  width.to_int()
}

///|
fn paged_blueprint_step(
  blueprint : PagedKvExecutionBlueprint,
  operation : @plan.OperationId,
) -> PagedKvExecutionStep raise DeviceStepError {
  let mut found : PagedKvExecutionStep? = None
  for step in blueprint.steps() {
    if step.operation_id() == operation {
      if found is Some(_) {
        raise executor_operation_error(operation, GraphBucket)
      }
      found = Some(step)
    }
  }
  guard found is Some(step) else {
    raise executor_operation_error(operation, GraphBucket)
  }
  step
}

///|
fn prepare_projection_bucket_launches(
  blueprint : PagedKvExecutionBlueprint,
  steps : ReadOnlyArray[PreparedPagedGraphStep],
  launches : FixedArray[PagedBucketStepLaunch],
) -> Unit raise DeviceStepError {
  let mlp_occurrences : Array[Int] = []
  for index, prepared in steps {
    let step = paged_blueprint_step(blueprint, prepared.operation_id)
    let source = step.dimensions()
    match step.operation_kind() {
      QkvProjection | OutputProjection => {
        let output_width = paged_projection_output_width(blueprint, step)
        launches[index] = if source.block_x() == 256 &&
          source.shared_memory_bytes() == 8192 {
          ProjectionTensorCoreGridX(source, output_width)
        } else {
          ProjectionScalarGridX(source, output_width)
        }
      }
      LanguageModelHead => ()
      GatedMlp => {
        let workspace = step
          .operands()
          .any(operand => operand.role() == Workspace)
        if workspace {
          let mut occurrence = 0
          for operation in mlp_occurrences {
            if operation == prepared.operation_id.as_int() {
              occurrence += 1
            }
          }
          if occurrence > 1 {
            raise executor_operation_error(prepared.operation_id, GraphBucket)
          }
          mlp_occurrences.push(prepared.operation_id.as_int())
          launches[index] = if occurrence == 0 {
            ProjectionMlpUpGridX(
              source,
              paged_projection_intermediate_width(blueprint, step),
            )
          } else {
            ProjectionMlpDownGridX(
              source,
              paged_projection_output_width(blueprint, step),
            )
          }
        } else if source.shared_memory_bytes() > 0 {
          launches[index] = QueryTokenCappedGridX(source)
        } else {
          launches[index] = ProjectionScalarGridX(
            source,
            paged_projection_output_width(blueprint, step),
          )
        }
      }
      _ => ()
    }
  }
}

///|
fn paged_qwen_postprocess_bucket_launch(
  source : @launch_contract.AotLaunchDimensions,
) -> PagedBucketStepLaunch {
  QueryTokenCappedGridX(source)
}

///|
/// Derives the only currently safe smaller-bucket launch contract from the
/// approved Qwen composite ABI. The V2 reusable postprocess and readonly
/// attention kernels both authenticate a bounded grid and use grid-stride token
/// traversal. Full ingress maps one block to a 16-token tile; the legacy
/// postprocess maps one block to one token.
/// Projection rules use their admitted shape-specific decode/prefill geometry,
/// avoiding maximum-envelope decode work where the kernel ABI permits it.
fn prepare_qwen_graph_launch_contract(
  blueprint : PagedKvExecutionBlueprint,
  steps : ReadOnlyArray[PreparedPagedGraphStep],
  spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
  prefill_split_steps? : ReadOnlyArray[PreparedPagedGraphStep]? = None,
) -> PagedGraphLaunchContract? raise DeviceStepError {
  if spans.is_empty() {
    return None
  }
  let launches = FixedArray::make(steps.length(), FixedBucketLaunch)
  let decode_launches = FixedArray::make(steps.length(), FixedBucketLaunch)
  prepare_projection_bucket_launches(blueprint, steps, launches)
  prepare_projection_bucket_launches(blueprint, steps, decode_launches)
  for span in spans {
    let (qkv_index, qk_norm_index, _, attention_index) = span.operation_indices()
    let ingress_index = if span.ingress_includes_projection() {
      qkv_index
    } else {
      qk_norm_index
    }
    guard blueprint.steps().get(ingress_index) is Some(ingress) &&
      blueprint.steps().get(attention_index) is Some(attention) else {
      raise InvalidExecutor(GraphBucket)
    }
    let postprocess = span.postprocess_launch()
    let readonly_launch = span.readonly_launch()
    let expected_ingress_grid = if span.ingress_includes_projection() {
      (blueprint.profile().max_query_tokens() + 15) / 16
    } else {
      Int::min(blueprint.profile().max_query_tokens(), 4096)
    }
    if postprocess.grid_x() != expected_ingress_grid ||
      readonly_launch.grid_x() <= 0 ||
      readonly_launch.grid_x() > blueprint.profile().max_query_tokens() {
      raise InvalidExecutor(GraphBucket)
    }
    let postprocess_index = qwen_bucket_step_index(
      steps,
      ingress.operation_id(),
    )
    let readonly_index = qwen_bucket_step_index(steps, attention.operation_id())
    match
      (
        launches[postprocess_index],
        launches[readonly_index],
        decode_launches[postprocess_index],
        decode_launches[readonly_index],
      ) {
      (_, FixedBucketLaunch, _, FixedBucketLaunch) => {
        launches[postprocess_index] = if span.ingress_includes_projection() {
          QueryTileCappedGridX(postprocess, 16)
        } else {
          paged_qwen_postprocess_bucket_launch(postprocess)
        }
        launches[readonly_index] = if span.has_phase_specific_attention() {
          QueryTileCappedGridX(readonly_launch, 16)
        } else {
          QueryTokenCappedGridX(readonly_launch)
        }
        decode_launches[postprocess_index] = if span.ingress_includes_projection() {
          QueryTileCappedGridX(postprocess, 16)
        } else {
          paged_qwen_postprocess_bucket_launch(postprocess)
        }
        decode_launches[readonly_index] = if span.has_phase_specific_attention() {
          QueryTokenCappedGridX(span.decode_launch())
        } else {
          QueryTokenCappedGridX(readonly_launch)
        }
      }
      _ => raise InvalidExecutor(GraphBucket)
    }
  }
  let prefill_split_step_launches = match prefill_split_steps {
    Some(selected) => {
      let split_launches = FixedArray::make(
        selected.length(),
        FixedBucketLaunch,
      )
      prepare_projection_bucket_launches(blueprint, selected, split_launches)
      for span in spans {
        let (qkv_index, qk_norm_index, _, attention_index) = span.operation_indices()
        let ingress_index = if span.ingress_includes_projection() {
          qkv_index
        } else {
          qk_norm_index
        }
        guard blueprint.steps().get(ingress_index) is Some(ingress) &&
          blueprint.steps().get(attention_index) is Some(attention) &&
          span.partitioned_prefill() is Some(pipeline) else {
          raise InvalidExecutor(GraphBucket)
        }
        let ingress_step = qwen_bucket_step_index(
          selected,
          ingress.operation_id(),
        )
        let attention_steps = qwen_bucket_step_indices(
          selected,
          attention.operation_id(),
        )
        if attention_steps.length() != 2 {
          raise InvalidExecutor(GraphBucket)
        }
        split_launches[ingress_step] = if span.ingress_includes_projection() {
          QueryTileCappedGridX(span.postprocess_launch(), 16)
        } else {
          QueryTokenCappedGridX(span.postprocess_launch())
        }
        split_launches[attention_steps[0]] = QueryTileCappedGridX(
          pipeline.partial_launch,
          16,
        )
        split_launches[attention_steps[1]] = QueryTileCappedGridX(
          pipeline.merge_launch,
          16,
        )
      }
      Some(split_launches)
    }
    None => None
  }
  Some({
    step_launches: launches,
    decode_step_launches: decode_launches,
    prefill_split_step_launches,
  })
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
fn paged_ceil_div(value : Int, divisor : Int) -> Int raise DeviceStepError {
  if value <= 0 || divisor <= 0 || value > 2147483647 - (divisor - 1) {
    raise InvalidExecutor(GraphBucket)
  }
  (value + divisor - 1) / divisor
}

///|
fn paged_projection_bucket_grid_x(
  rule : PagedBucketStepLaunch,
  query_token_bound : Int,
) -> Int raise DeviceStepError {
  if query_token_bound <= 0 {
    raise InvalidExecutor(GraphBucket)
  }
  match rule {
    FixedBucketLaunch => raise InvalidExecutor(GraphBucket)
    QueryTokenCappedGridX(source) =>
      Int::min(source.grid_x(), query_token_bound)
    QueryTileCappedGridX(source, query_tile) =>
      Int::min(source.grid_x(), paged_ceil_div(query_token_bound, query_tile))
    ProjectionTensorCoreGridX(source, output_width) => {
      let grid = if query_token_bound == 1 {
        paged_ceil_div(output_width, 8)
      } else {
        paged_ceil_div(output_width, 128) *
        paged_ceil_div(query_token_bound, 16)
      }
      Int::min(source.grid_x(), grid)
    }
    ProjectionScalarGridX(source, output_width) => {
      if query_token_bound > 2147483647 / output_width {
        raise InvalidExecutor(GraphBucket)
      }
      Int::min(
        source.grid_x(),
        paged_ceil_div(query_token_bound * output_width, source.block_x()),
      )
    }
    ProjectionMlpUpGridX(source, intermediate_width) => {
      let grid = if query_token_bound == 1 {
        paged_ceil_div(intermediate_width, 16)
      } else {
        paged_ceil_div(query_token_bound, 16) *
        paged_ceil_div(intermediate_width, 256)
      }
      Int::min(source.grid_x(), grid)
    }
    ProjectionMlpDownGridX(source, output_width) => {
      let grid = if query_token_bound == 1 {
        paged_ceil_div(output_width, 16)
      } else {
        paged_ceil_div(query_token_bound, 16) * paged_ceil_div(output_width, 64)
      }
      Int::min(source.grid_x(), grid)
    }
  }
}

///|
fn paged_bucket_launch(
  rule : PagedBucketStepLaunch,
  query_token_bound : Int,
) -> @launch_contract.AotLaunchDimensions raise DeviceStepError {
  if rule is QueryTokenCappedGridX(source) {
    return paged_capped_query_token_launch(source, query_token_bound)
  }
  if rule is QueryTileCappedGridX(source, query_tile) {
    return paged_capped_query_token_launch(
      source,
      paged_ceil_div(query_token_bound, query_tile),
    )
  }
  let source = match rule {
    FixedBucketLaunch => raise InvalidExecutor(GraphBucket)
    QueryTokenCappedGridX(value)
    | QueryTileCappedGridX(value, _)
    | ProjectionTensorCoreGridX(value, _)
    | ProjectionScalarGridX(value, _)
    | ProjectionMlpUpGridX(value, _)
    | ProjectionMlpDownGridX(value, _) => value
  }
  paged_scaled_query_token_launch(
    source,
    paged_projection_bucket_grid_x(rule, query_token_bound),
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
  phase_decode? : Bool = false,
  phase_prefill_split? : Bool = false,
) -> Array[@device.OrderedKernelLaunch] raise DeviceStepError {
  let launches : Array[@device.OrderedKernelLaunch] = []
  for index, step in steps {
    guard resources.functions.get(step.function_index) is Some(function) &&
      function.open else {
      raise InvalidExecutor(ResourceState)
    }
    let dimensions = match
      (query_token_bound, resources.graph_launch_contract) {
      (Some(bound), Some(contract)) => {
        let launch_contract = if phase_prefill_split {
          match contract.prefill_split_step_launches {
            Some(split) => split
            None => contract.step_launches
          }
        } else if phase_decode {
          contract.decode_step_launches
        } else {
          contract.step_launches
        }
        match launch_contract.get(index) {
          Some(FixedBucketLaunch) | None => step.dimensions
          Some(rule) =>
            paged_device_launch_dimensions(paged_bucket_launch(rule, bound))
        }
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
  prefill_split_steps? : ReadOnlyArray[PreparedPagedGraphStep]? = None,
  prefill_split_minimum_context_tokens? : Int = PAGED_DECODE_SPLIT_DISABLED_CONTEXT_THRESHOLD,
  decode_steps? : ReadOnlyArray[PreparedPagedGraphStep]? = None,
  decode_split_steps? : ReadOnlyArray[PreparedPagedGraphStep]? = None,
  decode_split_minimum_context_tokens? : Int = PAGED_DECODE_SPLIT_DISABLED_CONTEXT_THRESHOLD,
) -> PagedOrderedExecutorDispatch raise DeviceStepError {
  let maximum_launches = paged_ordered_launches(steps, resources)
  let baseline_operation_ids = ReadOnlyArray::from_array(
    steps.map(step => step.operation_id),
  )
  let slot_owner_indices = FixedArray::make(table.bucket_count(), -1)
  let prefill_split_slot_owner_indices = FixedArray::make(
    table.bucket_count(),
    -1,
  )
  let mut fallback_owner_index = -1
  let telemetry_owner_index = match graph_policy {
    PagedOrderedEagerOnly => {
      if resources.graph_launch_contract is Some(_) {
        for bound in paged_prefill_capture_bounds(profile) {
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
            OrderedKernelEagerOnly,
          )
          map_paged_prefill_capture_owner(
            table,
            profile.max_query_rows(),
            bound,
            owner,
            slot_owner_indices,
          )
        }
        if decode_steps is None {
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
              OrderedKernelEagerOnly,
            )
            map_paged_decode_capture_owner(
              table, bound, bound, owner, slot_owner_indices,
            )
          }
        }
      }
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
      // Qwen's composite launch contract can safely specialize fixed graph
      // geometry while retaining the same reusable descriptor and allocation
      // addresses. Capture the common decode buckets up front so c1/c8 do not
      // replay the maximum-token grid full of empty blocks. The maximum graph
      // remains the universal fallback for prefill, mixed, and uncommon decode
      // shapes.
      if resources.graph_launch_contract is Some(_) {
        for bound in paged_prefill_capture_bounds(profile) {
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
          map_paged_prefill_capture_owner(
            table,
            profile.max_query_rows(),
            bound,
            owner,
            slot_owner_indices,
          )
        }
        if decode_steps is None {
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
      // Capture-with-fallback reports OrderedKernelEager from this same owner
      // when the backend lacks graph support; no duplicate launch list is
      // needed. Non-maximum buckets therefore use the selected maximum owner
      // in either backend mode.
      fallback_owner_index = maximum
      maximum
    }
  }
  let prefill_split_owner_index = match prefill_split_steps {
    Some(selected) => {
      let operation_ids = ReadOnlyArray::from_array(
        selected.map(step => step.operation_id),
      )
      if resources.graph_launch_contract is Some(_) {
        for bound in paged_prefill_capture_bounds(profile) {
          let launches = paged_ordered_launches(
            selected,
            resources,
            query_token_bound=Some(bound),
            phase_prefill_split=true,
          )
          let owner = prepare_one_paged_ordered_executor(
            context,
            launches[:],
            operation_ids,
            resources,
            paged_ordered_device_policy(graph_policy),
          )
          map_paged_prefill_capture_owner(
            table,
            profile.max_query_rows(),
            bound,
            owner,
            prefill_split_slot_owner_indices,
          )
        }
      }
      let launches = paged_ordered_launches(selected, resources)
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
  let decode_owner_index = match decode_steps {
    Some(selected) => {
      let operation_ids = ReadOnlyArray::from_array(
        selected.map(step => step.operation_id),
      )
      if resources.graph_launch_contract is Some(_) {
        clear_paged_decode_capture_owners(
          table,
          profile.max_query_rows(),
          profile.max_query_rows(),
          slot_owner_indices,
        )
        for bound in paged_decode_capture_bounds(profile) {
          let launches = paged_ordered_launches(
            selected,
            resources,
            query_token_bound=Some(bound),
            phase_decode=true,
          )
          let owner = prepare_one_paged_ordered_executor(
            context,
            launches[:],
            operation_ids,
            resources,
            paged_ordered_device_policy(graph_policy),
          )
          map_paged_decode_capture_owner(
            table, bound, bound, owner, slot_owner_indices,
          )
        }
      }
      let maximum_launches = paged_ordered_launches(selected, resources)
      prepare_one_paged_ordered_executor(
        context,
        maximum_launches[:],
        operation_ids,
        resources,
        paged_ordered_device_policy(graph_policy),
      )
    }
    None => -1
  }
  let decode_split_owner_index = match decode_split_steps {
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
    prefill_split_slot_owner_indices,
    fallback_owner_index,
    telemetry_owner_index,
    prefill_split_owner_index,
    prefill_split_minimum_context_tokens,
    decode_owner_index,
    decode_split_owner_index,
    decode_split_minimum_context_tokens,
  }
}
