///|
fn qwen_fused_postprocess_argument_count() -> Int {
  12
}

///|
fn qwen_fused_full_ingress_argument_count() -> Int {
  15
}

///|
priv struct PreparedQwenFusedAttentionResources {
  ingress_function : Int
  prefill_function : Int
  wide_prefill_function : Int?
  decode_function : Int
  sidecar : @device.Allocation
}

///|
fn qwen_fused_attention_sidecar_shape_from_bindings(
  rotated : PagedKvOperandBinding,
  readonly_input : PagedKvOperandBinding,
) -> (Int64, Int64) raise DeviceStepError {
  if !same_composite_region(rotated, readonly_input) ||
    rotated.byte_count() <= 0L ||
    rotated.alignment() <= 0L {
    raise InvalidExecutor(ResourceState)
  }
  (rotated.byte_count(), rotated.alignment())
}

///|
fn qwen_fused_attention_sidecar_shape(
  blueprint : PagedKvExecutionBlueprint,
  span : ApprovedQwenFusedAttentionSpan,
) -> (Int64, Int64) raise DeviceStepError {
  let (_, _, rope_index, attention_index) = span.operation_indices()
  guard blueprint.steps().get(rope_index) is Some(rope_step) &&
    blueprint.steps().get(attention_index) is Some(attention_step) else {
    raise InvalidExecutor(ResourceState)
  }
  let rotated = unique_composite_binding(rope_step, role => {
    role is ActivationOutput(_)
  })
  let readonly_input = unique_composite_binding(attention_step, role => {
    role is ActivationInput(_)
  })
  qwen_fused_attention_sidecar_shape_from_bindings(rotated, readonly_input)
}

///|
fn prepare_qwen_fused_attention_resources(
  context : @device.Context,
  blueprint : PagedKvExecutionBlueprint,
  spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
  resources : PagedGraphResources,
) -> PreparedQwenFusedAttentionResources raise DeviceStepError {
  guard spans.get(0) is Some(first) else {
    raise InvalidExecutor(ResourceState)
  }
  let sidecar_shape = qwen_fused_attention_sidecar_shape(blueprint, first)
  for span in spans {
    if !span.same_runtime_modules(first) ||
      qwen_fused_attention_sidecar_shape(blueprint, span) != sidecar_shape {
      raise InvalidExecutor(ResourceState)
    }
  }
  let postprocess = prepare_composite_function(
    context,
    first.postprocess_module_bytes(),
    first.postprocess_function_symbol(),
    resources,
  )
  let readonly_function = prepare_composite_function(
    context,
    first.readonly_module_bytes(),
    first.readonly_function_symbol(),
    resources,
  )
  let decode_function = if first.has_phase_specific_attention() {
    prepare_composite_function(
      context,
      first.decode_module_bytes(),
      first.decode_function_symbol(),
      resources,
    )
  } else {
    readonly_function
  }
  let wide_prefill_function = match first.wide_query_prefill() {
    Some(variant) =>
      Some(
        prepare_composite_function(
          context,
          variant.module_bytes,
          variant.symbol,
          resources,
        ),
      )
    None => None
  }
  if resources.qwen_fused_attention_sidecar is Some(_) {
    raise InvalidExecutor(ResourceState)
  }
  let sidecar = context.allocate(sidecar_shape.0) catch {
    error => raise ExecutorDevice(ActivationWorkspaceArena, error)
  }
  resources.qwen_fused_attention_sidecar = Some({
    resource: sidecar,
    open: true,
  })
  if sidecar.byte_count() != sidecar_shape.0 {
    raise InvalidExecutor(AllocationSize)
  }
  context.validate_allocation_region(
    sidecar,
    offset=0L,
    byte_count=sidecar_shape.0,
    alignment=sidecar_shape.1,
  ) catch {
    error => raise ExecutorDevice(KernelArguments, error)
  }
  {
    ingress_function: postprocess,
    prefill_function: readonly_function,
    wide_prefill_function,
    decode_function,
    sidecar,
  }
}

///|
fn qwen_fused_attention_weights(
  step : PagedKvExecutionStep,
  expected : Int,
) -> Array[PagedKvOperandBinding] raise DeviceStepError {
  let weights : Array[PagedKvOperandBinding] = []
  for binding in step.operands() {
    if binding.role() is WeightInput(_) {
      weights.push(binding)
    }
  }
  if weights.length() != expected {
    raise executor_operation_error(step.operation_id(), ResourceState)
  }
  weights
}

///|
fn qwen_fused_sidecar_argument(
  binding : PagedKvOperandBinding,
  sidecar : @device.Allocation,
) -> @device.KernelArgument raise DeviceStepError {
  @device.KernelArgument::device_region(
    sidecar,
    offset=0L,
    byte_count=binding.byte_count(),
    alignment=binding.alignment(),
  ) catch {
    error => raise ExecutorDevice(KernelArguments, error)
  }
}

///|
fn prepare_qwen_fused_attention_launch(
  operation : @plan.OperationId,
  function_index : Int,
  launch : @launch_contract.AotLaunchDimensions,
  bindings : ArrayView[PagedKvOperandBinding],
  sidecar_binding_index : Int,
  sidecar : @device.Allocation,
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation : @device.Allocation,
  kv : @device.Allocation,
) -> PreparedPagedGraphStep raise DeviceStepError {
  if sidecar_binding_index < 0 || sidecar_binding_index >= bindings.length() {
    raise executor_operation_error(operation, ResourceState)
  }
  let arguments : Array[@device.KernelArgument] = []
  for index, binding in bindings {
    arguments.push(
      if index == sidecar_binding_index {
        qwen_fused_sidecar_argument(binding, sidecar)
      } else {
        composite_binding_argument(binding, descriptor, weight, activation, kv)
      },
    )
  }
  let packed = @device.KernelArguments::new(arguments) catch {
    error => raise ExecutorDevice(KernelArguments, error)
  }
  {
    operation_id: operation,
    function_index,
    dimensions: composite_launch_dimensions(launch, operation),
    arguments: packed,
  }
}

///|
fn prepare_qwen_fused_attention_steps(
  blueprint : PagedKvExecutionBlueprint,
  span : ApprovedQwenFusedAttentionSpan,
  ingress_function : Int,
  readonly_function : Int,
  readonly_launch : @launch_contract.AotLaunchDimensions,
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation : @device.Allocation,
  kv : @device.Allocation,
  sidecar : @device.Allocation,
) -> (PreparedPagedGraphStep, PreparedPagedGraphStep) raise DeviceStepError {
  let (qkv_index, qk_norm_index, rope_index, attention_index) = span.operation_indices()
  guard blueprint.steps().get(qkv_index) is Some(qkv_step) &&
    blueprint.steps().get(qk_norm_index) is Some(qk_norm_step) &&
    blueprint.steps().get(rope_index) is Some(rope_step) &&
    blueprint.steps().get(attention_index) is Some(attention_step) else {
    raise InvalidExecutor(ResourceState)
  }
  let attention_counts = unique_composite_binding(attention_step, role => {
    role == RuntimeInput(StepCounts)
  })
  for step in [qkv_step, qk_norm_step, rope_step] {
    let counts = unique_composite_binding(step, role => {
      role == RuntimeInput(StepCounts)
    })
    if !same_composite_region(counts, attention_counts) {
      raise InvalidExecutor(ResourceState)
    }
  }
  let query_positions = unique_composite_binding(attention_step, role => {
    role == RuntimeInput(QueryPositions)
  })
  let rope_positions = unique_composite_binding(rope_step, role => {
    role == RuntimeInput(QueryPositions)
  })
  if !same_composite_region(rope_positions, query_positions) {
    raise InvalidExecutor(ResourceState)
  }
  let runtime = [
    attention_counts,
    query_positions,
    unique_composite_binding(attention_step, role => {
      role == RuntimeInput(QueryRowOffsets)
    }),
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
  let qkv_output = unique_composite_binding(qkv_step, role => {
    role is ActivationOutput(_)
  })
  let qk_norm_input = unique_composite_binding(qk_norm_step, role => {
    role is ActivationInput(_)
  })
  let qk_norm_output = unique_composite_binding(qk_norm_step, role => {
    role is ActivationOutput(_)
  })
  let rope_input = unique_composite_binding(rope_step, role => {
    role is ActivationInput(_)
  })
  let rotated = unique_composite_binding(rope_step, role => {
    role is ActivationOutput(_)
  })
  let readonly_input = unique_composite_binding(attention_step, role => {
    role is ActivationInput(_)
  })
  if !same_composite_region(qkv_output, qk_norm_input) ||
    !same_composite_region(qk_norm_output, rope_input) ||
    !same_composite_region(rotated, readonly_input) {
    raise InvalidExecutor(ResourceState)
  }
  let qk_norm_weights = qwen_fused_attention_weights(qk_norm_step, 2)
  let qkv_weights = qwen_fused_attention_weights(qkv_step, 3)
  let output = unique_composite_binding(attention_step, role => {
    role is ActivationOutput(_)
  })
  let key_cache = unique_composite_binding(attention_step, role => {
    role is PersistentKvComponent(_, Key)
  })
  let value_cache = unique_composite_binding(attention_step, role => {
    role is PersistentKvComponent(_, Value)
  })
  let postprocess_bindings : Array[PagedKvOperandBinding] = []
  for binding in runtime {
    postprocess_bindings.push(binding)
  }
  if span.ingress_includes_projection() {
    postprocess_bindings.push(
      unique_composite_binding(qkv_step, role => role is ActivationInput(_)),
    )
    for binding in qkv_weights {
      postprocess_bindings.push(binding)
    }
  } else {
    postprocess_bindings.push(qkv_output)
  }
  for binding in qk_norm_weights {
    postprocess_bindings.push(binding)
  }
  for binding in [rotated, key_cache, value_cache] {
    postprocess_bindings.push(binding)
  }
  let readonly_bindings : Array[PagedKvOperandBinding] = []
  for binding in runtime {
    readonly_bindings.push(binding)
  }
  for binding in [readonly_input, output, key_cache, value_cache] {
    readonly_bindings.push(binding)
  }
  let expected_ingress_arguments = if span.ingress_includes_projection() {
    qwen_fused_full_ingress_argument_count()
  } else {
    qwen_fused_postprocess_argument_count()
  }
  if postprocess_bindings.length() != expected_ingress_arguments ||
    readonly_bindings.length() != readonly_attention_production_argument_count() {
    raise InvalidExecutor(ResourceState)
  }
  (
    prepare_qwen_fused_attention_launch(
      if span.ingress_includes_projection() {
        qkv_step.operation_id()
      } else {
        qk_norm_step.operation_id()
      },
      ingress_function,
      span.postprocess_launch(),
      postprocess_bindings,
      postprocess_bindings.length() - 3,
      sidecar,
      descriptor,
      weight,
      activation,
      kv,
    ),
    prepare_qwen_fused_attention_launch(
      attention_step.operation_id(),
      readonly_function,
      readonly_launch,
      readonly_bindings,
      readonly_bindings.length() - 4,
      sidecar,
      descriptor,
      weight,
      activation,
      kv,
    ),
  )
}

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
  function_index : Int,
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
) -> ReadOnlyArray[PreparedPagedGraphStep]? raise DeviceStepError {
  guard spans.get(0) is Some(first) else { return None }
  guard first.partitioned_prefill() is Some(pipeline) else { return None }
  for span in spans {
    if span.partitioned_prefill() != Some(pipeline) {
      raise InvalidExecutor(ResourceState)
    }
  }
  if resources.prefill_partition_workspace is Some(_) ||
    pipeline.workspace_bytes <= 0L ||
    pipeline.workspace_bytes > 1073741824L {
    raise InvalidExecutor(ResourceState)
  }
  let workspace = context.allocate(pipeline.workspace_bytes) catch {
    error => raise ExecutorDevice(DecodeSplitWorkspace, error)
  }
  resources.prefill_partition_workspace = Some({
    resource: workspace,
    open: true,
  })
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
    (Int, PreparedPagedGraphStep, PreparedPagedGraphStep),
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
        composite_binding_argument(binding, descriptor, weight, activation, kv),
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
) -> Int raise DeviceStepError {
  guard spans.get(0) is Some(first) else {
    return PAGED_DECODE_SPLIT_DISABLED_CONTEXT_THRESHOLD
  }
  guard first.partitioned_prefill() is Some(pipeline) else {
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
/// Returns the query tile used to compile the low-parallelism partition map.
/// The partitioned compiler intentionally receives exactly one query tile, so
/// its admitted grid-X is also the immutable tile extent. Runtime dispatch
/// uses this backend-neutral extent to avoid adding KV partitions only for a
/// single row whose query map already exposes independent tiles. Batched and
/// mixed rows retain the partitioned reduction until the baseline kernel's
/// cross-row numerical contract is independently established.
fn qwen_partitioned_prefill_query_tile(
  spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
) -> Int raise DeviceStepError {
  guard spans.get(0) is Some(first) &&
    first.partitioned_prefill() is Some(first_pipeline) else {
    return 1
  }
  let query_tile = first_pipeline.partial_launch.grid_x()
  if query_tile <= 0 {
    raise InvalidExecutor(ResourceState)
  }
  for span in spans {
    guard span.partitioned_prefill() is Some(pipeline) &&
      pipeline.partial_launch.grid_x() == query_tile else {
      raise InvalidExecutor(ResourceState)
    }
  }
  query_tile
}

///|
/// Returns the portable query-token threshold attached to the wide-query
/// frontier role. Every repeated Qwen layer must share the same immutable
/// variant; request dispatch then compares one bucket bound without scanning.
fn qwen_wide_prefill_minimum_query_tokens(
  spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
) -> Int raise DeviceStepError {
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

///|
fn[Step] replace_qwen_fused_attention_runtime_spans(
  baseline : ReadOnlyArray[Step],
  spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
  postprocess : ReadOnlyArray[Step],
  readonly_steps : ReadOnlyArray[Step],
) -> ReadOnlyArray[Step] raise DeviceStepError {
  if spans.length() != postprocess.length() ||
    spans.length() != readonly_steps.length() {
    raise InvalidExecutor(ResourceState)
  }
  if spans.all(span => !span.ingress_includes_projection()) {
    let qkv_indices : Array[Int] = []
    for span in spans {
      qkv_indices.push(span.operation_indices().0)
    }
    return replace_projection_postprocess_attention_spans(
      baseline,
      ReadOnlyArray::from_array(qkv_indices),
      postprocess,
      readonly_steps,
    )
  }
  let replacement_for = FixedArray::make(baseline.length(), -1)
  let skipped = FixedArray::make(baseline.length(), false)
  for replacement_index, span in spans {
    let indices = span.operation_indices()
    let start = if span.ingress_includes_projection() {
      indices.0
    } else {
      indices.1
    }
    if indices.0 < 0 ||
      indices.3 >= baseline.length() ||
      start < indices.0 ||
      start > indices.1 ||
      replacement_for[start] >= 0 {
      raise InvalidExecutor(ResourceState)
    }
    for index in start..<=indices.3 {
      if skipped[index] {
        raise InvalidExecutor(ResourceState)
      }
      skipped[index] = true
    }
    replacement_for[start] = replacement_index
  }
  let output : Array[Step] = []
  for index, step in baseline {
    let replacement = replacement_for[index]
    if replacement >= 0 {
      output.push(postprocess[replacement])
      output.push(readonly_steps[replacement])
    } else if !skipped[index] {
      output.push(step)
    }
  }
  ReadOnlyArray::from_array(output)
}

///|
fn qwen_fused_attention_overlaps_residual(
  qwen_spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
  residual_index : Int,
  norm_index : Int,
) -> Bool {
  for span in qwen_spans {
    let indices = span.operation_indices()
    if residual_index == indices.0 ||
      residual_index == indices.1 ||
      residual_index == indices.2 ||
      residual_index == indices.3 ||
      norm_index == indices.0 ||
      norm_index == indices.1 ||
      norm_index == indices.2 ||
      norm_index == indices.3 {
      return true
    }
  }
  false
}
