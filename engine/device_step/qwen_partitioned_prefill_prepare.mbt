///|
fn qwen_partitioned_prefill_workspace_argument(
  workspace : @device.Allocation,
  bytes : Int64,
  alignment : Int64,
) -> @device.KernelArgument raise DeviceStepError {
  @device.KernelArgument::device_region(
    workspace,
    offset=0L,
    byte_count=bytes,
    alignment~,
  ) catch {
    error => raise ExecutorDevice(KernelArguments, error)
  }
}

///|
fn qwen_partitioned_prefill_step(
  operation : @plan.OperationId,
  function_index : Int32,
  launch : @launch_contract.AotLaunchDimensions,
  arguments : ArrayView[@device.KernelArgument],
) -> PreparedPagedGraphStep raise DeviceStepError {
  {
    operation_id: operation,
    function_index,
    dimensions: composite_launch_dimensions(launch, operation),
    arguments: @device.KernelArguments::new(arguments) catch {
      error => raise ExecutorDevice(KernelArguments, error)
    },
  }
}

///|
fn qwen_partitioned_prefill_variant(
  span : ApprovedQwenFusedAttentionSpan,
  deep : Bool,
) -> ApprovedQwenPartitionedPrefill? {
  if deep {
    span.deep_partitioned_prefill()
  } else {
    span.partitioned_prefill()
  }
}

///|
fn prepare_qwen_partitioned_prefill_steps(
  context : @device.Context,
  blueprint : PagedKvExecutionBlueprint,
  baseline : ReadOnlyArray[PreparedPagedGraphStep],
  spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation : @device.Allocation,
  kv : @device.Allocation,
  resources : PagedGraphResources,
  deep~ : Bool = false,
) -> ReadOnlyArray[PreparedPagedGraphStep]? raise DeviceStepError {
  guard spans.get(0) is Some(first) else { return None }
  guard qwen_partitioned_prefill_variant(first, deep) is Some(pipeline) else {
    return None
  }
  for span in spans {
    if qwen_partitioned_prefill_variant(span, deep) != Some(pipeline) {
      raise InvalidExecutor(ResourceState)
    }
  }
  let workspace_present = if deep {
    resources.deep_prefill_partition_workspace is Some(_)
  } else {
    resources.prefill_partition_workspace is Some(_)
  }
  if workspace_present ||
    pipeline.workspace_bytes <= 0L ||
    pipeline.workspace_bytes > 1073741824L {
    raise InvalidExecutor(ResourceState)
  }
  let workspace = context.allocate(pipeline.workspace_bytes) catch {
    error => raise ExecutorDevice(DecodeSplitWorkspace, error)
  }
  if deep {
    resources.deep_prefill_partition_workspace = Some({
      resource: workspace,
      open: true,
    })
  } else {
    resources.prefill_partition_workspace = Some({
      resource: workspace,
      open: true,
    })
  }
  context.validate_allocation_region(
    workspace,
    offset=0L,
    byte_count=pipeline.workspace_bytes,
    alignment=pipeline.workspace_alignment,
  ) catch {
    error => raise ExecutorDevice(DecodeSplitWorkspace, error)
  }
  let partial_function = prepare_composite_function(
    context,
    pipeline.module_bytes,
    pipeline.partial_symbol,
    resources,
  )
  let merge_function = prepare_composite_function(
    context,
    pipeline.module_bytes,
    pipeline.merge_symbol,
    resources,
  )
  let replacements : Array[
    (Int32, PreparedPagedGraphStep, PreparedPagedGraphStep),
  ] = []
  for span in spans {
    let (_, _, rope_index, attention_index) = span.operation_indices()
    guard blueprint.steps().get(rope_index) is Some(rope_step) &&
      blueprint.steps().get(attention_index) is Some(attention_step) else {
      raise InvalidExecutor(ResourceState)
    }
    let counts = unique_composite_binding(attention_step, role => {
      role == RuntimeInput(StepCounts)
    })
    let row_offsets = unique_composite_binding(attention_step, role => {
      role == RuntimeInput(QueryRowOffsets)
    })
    let rotated = unique_composite_binding(rope_step, role => {
      role is ActivationOutput(_)
    })
    let key_cache = unique_composite_binding(attention_step, role => {
      role is PersistentKvComponent(_, Key)
    })
    let value_cache = unique_composite_binding(attention_step, role => {
      role is PersistentKvComponent(_, Value)
    })
    let output = unique_composite_binding(attention_step, role => {
      role is ActivationOutput(_)
    })
    guard resources.qwen_fused_attention_sidecar is Some(sidecar) &&
      sidecar.open else {
      raise InvalidExecutor(ResourceState)
    }
    let workspace_argument = qwen_partitioned_prefill_workspace_argument(
      workspace,
      pipeline.workspace_bytes,
      pipeline.workspace_alignment,
    )
    let runtime_bindings = [
      counts,
      unique_composite_binding(attention_step, role => {
        role == RuntimeInput(QueryPositions)
      }),
      row_offsets,
      unique_composite_binding(attention_step, role => {
        role == RuntimeInput(SequenceLengths)
      }),
      unique_composite_binding(attention_step, role => {
        role == RuntimeInput(PageTableOffsets)
      }),
      unique_composite_binding(attention_step, role => {
        role == RuntimeInput(PageTablePageIndices)
      }),
    ]
    let partial_arguments : Array[@device.KernelArgument] = []
    for binding in runtime_bindings {
      partial_arguments.push(
        if pipeline.query_metadata &&
          binding.role() == RuntimeInput(QueryRowOffsets) {
          attention_metadata_argument(descriptor, blueprint.profile())
        } else {
          composite_binding_argument(
            binding, descriptor, weight, activation, kv,
          )
        },
      )
    }
    partial_arguments.push(
      qwen_fused_sidecar_argument(rotated, sidecar.resource),
    )
    for binding in [key_cache, value_cache] {
      partial_arguments.push(
        composite_binding_argument(binding, descriptor, weight, activation, kv),
      )
    }
    partial_arguments.push(workspace_argument)
    let merge_arguments : Array[@device.KernelArgument] = [
      composite_binding_argument(counts, descriptor, weight, activation, kv),
      composite_binding_argument(
        row_offsets, descriptor, weight, activation, kv,
      ),
      workspace_argument,
      composite_binding_argument(output, descriptor, weight, activation, kv),
    ]
    replacements.push(
      (
        attention_step.operation_id().as_int(),
        qwen_partitioned_prefill_step(
          attention_step.operation_id(),
          partial_function,
          pipeline.partial_launch,
          partial_arguments,
        ),
        qwen_partitioned_prefill_step(
          attention_step.operation_id(),
          merge_function,
          pipeline.merge_launch,
          merge_arguments,
        ),
      ),
    )
    if span.has_phase_specific_attention() {
      let decode = paged_attention_decode_companion(
        resources,
        attention_step.operation_id(),
      )
      guard decode is Some(_) else { raise InvalidExecutor(GraphBucket) }
      partial_arguments[0] = prefill_attention_counts_argument(descriptor)
      merge_arguments[0] = prefill_attention_counts_argument(descriptor)
      register_attention_phase_replacement(
        resources,
        qwen_partitioned_prefill_step(
          attention_step.operation_id(),
          partial_function,
          pipeline.partial_launch,
          partial_arguments,
        ),
        None,
      )
      register_attention_phase_replacement(
        resources,
        qwen_partitioned_prefill_step(
          attention_step.operation_id(),
          merge_function,
          pipeline.merge_launch,
          merge_arguments,
        ),
        decode,
      )
    }
  }
  Some(
    replace_paged_decode_split_steps(
      baseline,
      ReadOnlyArray::from_array(replacements),
      step => step.operation_id.as_int(),
    ),
  )
}

///|
fn qwen_partitioned_prefill_minimum_context_tokens(
  blueprint : PagedKvExecutionBlueprint,
  spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
  deep~ : Bool = false,
) -> Int32 raise DeviceStepError {
  guard spans.get(0) is Some(first) else {
    return PAGED_DECODE_SPLIT_DISABLED_CONTEXT_THRESHOLD
  }
  guard qwen_partitioned_prefill_variant(first, deep) is Some(pipeline) else {
    return PAGED_DECODE_SPLIT_DISABLED_CONTEXT_THRESHOLD
  }
  let partitions = pipeline.partial_launch.grid_z()
  let tokens_per_page = blueprint.layout().tokens_per_page()
  if partitions <= 1 ||
    tokens_per_page <= 0 ||
    partitions > 2147483647 / 16 / tokens_per_page {
    raise InvalidExecutor(ResourceState)
  }
  partitions * 16 * tokens_per_page
}

///|
/// Returns the query-tile extent owned by the versioned runtime ABI. Grid-X is
/// launch coverage and is deliberately never reinterpreted as tile width.
fn qwen_partitioned_prefill_query_tile(
  blueprint : PagedKvExecutionBlueprint,
  spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
  deep~ : Bool = false,
) -> Int32 raise DeviceStepError {
  guard spans.get(0) is Some(first) &&
    qwen_partitioned_prefill_variant(first, deep) is Some(first_pipeline) else {
    return 1
  }
  let query_tile = first_pipeline.query_tile_rows
  if query_tile <= 0 || blueprint.profile().max_query_tokens() <= 0 {
    raise InvalidExecutor(ResourceState)
  }
  for span in spans {
    guard qwen_partitioned_prefill_variant(span, deep) is Some(pipeline) &&
      pipeline.query_tile_rows == query_tile else {
      raise InvalidExecutor(ResourceState)
    }
  }
  query_tile
}

///|
/// Returns the immutable number of associative KV partitions emitted by the
/// functional compiler. Bucket selection uses it as portable parallelism
/// metadata and never inspects a CUDA symbol on the token path.
fn qwen_partitioned_prefill_partition_count(
  spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
  deep~ : Bool = false,
) -> Int32 raise DeviceStepError {
  guard spans.get(0) is Some(first) &&
    qwen_partitioned_prefill_variant(first, deep) is Some(first_pipeline) else {
    return 1
  }
  let partitions = first_pipeline.partial_launch.grid_z()
  if partitions <= 1 {
    raise InvalidExecutor(ResourceState)
  }
  for span in spans {
    guard qwen_partitioned_prefill_variant(span, deep) is Some(pipeline) &&
      pipeline.partial_launch.grid_z() == partitions else {
      raise InvalidExecutor(ResourceState)
    }
  }
  partitions
}

///|
/// Returns the portable query-token threshold attached to the wide-query
/// frontier role. Every repeated Qwen layer must share the same immutable
/// variant; request dispatch then compares one bucket bound without scanning.
fn qwen_wide_prefill_minimum_query_tokens(
  spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
) -> Int32 raise DeviceStepError {
  guard spans.get(0) is Some(first) &&
    first.wide_query_prefill() is Some(first_variant) else {
    return PAGED_DECODE_SPLIT_DISABLED_CONTEXT_THRESHOLD
  }
  if first_variant.query_tile_rows <= 0 {
    raise InvalidExecutor(ResourceState)
  }
  for span in spans {
    guard span.wide_query_prefill() is Some(variant) &&
      variant.query_tile_rows == first_variant.query_tile_rows else {
      raise InvalidExecutor(ResourceState)
    }
  }
  first_variant.query_tile_rows
}
