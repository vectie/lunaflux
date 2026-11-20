///|
fn paged_attention_phase_route(
  phase : @luna_execution_graph_strategy.ExecutionGraphPhase,
  split_available : Bool,
) -> PagedAttentionPhaseRoute {
  if split_available && phase == DecodeGraph {
    DecodeSplitAttentionPhase
  } else {
    BaselineAttentionPhase
  }
}

///|
fn[T] replace_paged_decode_split_steps(
  baseline : ReadOnlyArray[T],
  replacements : ReadOnlyArray[(Int32, T, T)],
  operation : (T) -> Int32,
) -> ReadOnlyArray[T] raise DeviceStepError {
  let result : Array[T] = []
  let used = FixedArray::make(replacements.length(), false)
  for step in baseline {
    let mut replacement = -1
    for index, item in replacements {
      if item.0 == operation(step) {
        if replacement >= 0 {
          raise InvalidExecutor(ResourceState)
        }
        replacement = index
      }
    }
    if replacement < 0 {
      result.push(step)
    } else {
      result.push(replacements[replacement].1)
      result.push(replacements[replacement].2)
      used[replacement] = true
    }
  }
  if used.any(value => !value) {
    raise InvalidExecutor(ResourceState)
  }
  ReadOnlyArray::from_array(result)
}

///|
fn paged_decode_split_step(
  blueprint : PagedKvExecutionBlueprint,
  operation : @plan.OperationId,
) -> PagedKvExecutionStep raise DeviceStepError {
  let mut found : PagedKvExecutionStep? = None
  for step in blueprint.steps() {
    if step.operation_id() == operation {
      if found is Some(_) || step.operation_kind() != CausalAttention {
        raise executor_operation_error(operation, ResourceState)
      }
      found = Some(step)
    }
  }
  guard found is Some(step) else {
    raise executor_operation_error(operation, ResourceState)
  }
  step
}

///|
fn prepare_paged_decode_split_steps(
  context : @device.Context,
  blueprint : PagedKvExecutionBlueprint,
  baseline : ReadOnlyArray[PreparedPagedGraphStep],
  bindings : ReadOnlyArray[
    @luna_cuda_paged_attention_aot.PagedAttentionDecodeSplitRuntimeCudaBinding,
  ],
  maximum_workspace_bytes : Int64,
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation : @device.Allocation,
  kv : @device.Allocation,
  resources : PagedGraphResources,
  fused_qkv_readonly_span : ApprovedFusedQkvReadOnlyAttentionSpan?,
  qwen_spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
) -> ReadOnlyArray[PreparedPagedGraphStep]? raise DeviceStepError {
  if bindings.is_empty() {
    return None
  }
  let (workspace_bytes, workspace_alignment) = paged_decode_split_workspace_envelope(
    bindings, maximum_workspace_bytes,
  )
  validate_paged_decode_split_bindings(
    blueprint, bindings, workspace_bytes, workspace_alignment, fused_qkv_readonly_span,
    qwen_spans,
  )
  if resources.decode_split_workspace is Some(_) {
    raise InvalidExecutor(ResourceState)
  }
  let workspace = context.allocate(workspace_bytes) catch {
    error => raise ExecutorDevice(DecodeSplitWorkspace, error)
  }
  resources.decode_split_workspace = Some({ resource: workspace, open: true, })
  if workspace.byte_count() != workspace_bytes {
    raise InvalidExecutor(AllocationSize)
  }
  context.validate_allocation_region(
    workspace,
    offset=0L,
    byte_count=workspace_bytes,
    alignment=workspace_alignment,
  ) catch {
    error => raise ExecutorDevice(DecodeSplitWorkspace, error)
  }
  let loaded_modules : Array[PagedDecodeSplitLoadedModule] = []
  let loaded_functions : Array[PagedDecodeSplitLoadedFunction] = []
  let prepared : Array[PreparedPagedDecodeSplit] = []
  for binding in bindings {
    let operation = binding.operation_id()
    if binding.target() != blueprint.target() ||
      binding.profile() != blueprint.profile() ||
      binding.device_kv_layout() != blueprint.layout() {
      raise executor_operation_error(operation, ResourceState)
    }
    if prepared.any(item => item.operation_id == operation) {
      raise executor_operation_error(operation, ResourceState)
    }
    paged_decode_split_rejects_composite_overlap(
      blueprint, operation, fused_qkv_readonly_span, qwen_spans,
    )
    let step = paged_decode_split_step(blueprint, operation)
    let qwen_sidecar_shape = paged_decode_split_qwen_sidecar_shape(
      blueprint, operation, qwen_spans,
    )
    let qwen_sidecar = match qwen_sidecar_shape {
      Some(shape) => {
        guard resources.qwen_fused_attention_sidecar is Some(sidecar) &&
          sidecar.open &&
          sidecar.resource.byte_count() ==
          shape.0 +
          qwen_step_rotary_bytes_for_operation(blueprint, operation, qwen_spans) else {
          raise executor_operation_error(operation, ResourceState)
        }
        Some(sidecar.resource)
      }
      None => None
    }
    let module_index = paged_decode_split_loaded_module(
      context, binding, loaded_modules, resources,
    )
    let launches = binding.contract().launches()
    if launches.length() != 2 ||
      launches[0].stage() != Partial ||
      launches[1].stage() != Merge {
      raise executor_operation_error(operation, ResourceState)
    }
    let partial_function = paged_decode_split_loaded_function(
      module_index,
      launches[0].function_symbol(),
      loaded_functions,
      resources,
    )
    let merge_function = paged_decode_split_loaded_function(
      module_index,
      launches[1].function_symbol(),
      loaded_functions,
      resources,
    )
    let companion : PreparedPagedDecodeSplit = {
      operation_id: operation,
      partial_dimensions: launches[0].dimensions(),
      merge_dimensions: launches[1].dimensions(),
      partial: prepare_paged_decode_split_launch(
        operation,
        step,
        partial_function,
        launches[0],
        workspace,
        qwen_sidecar,
        descriptor,
        weight,
        activation,
        kv,
      ),
      merge: prepare_paged_decode_split_launch(
        operation,
        step,
        merge_function,
        launches[1],
        workspace,
        qwen_sidecar,
        descriptor,
        weight,
        activation,
        kv,
      ),
    }
    prepared.push(companion)
    // Read-only split kernels index decode rows after the prefill prefix.
    // Their scratch is already owned by this executor; mixed graphs reuse it
    // serially rather than allocating a second KV or workspace arena.
    if binding.is_blockwise_readonly() || binding.is_compiler_readonly() {
      resources.attention_partitioned_decode_companions.push(companion)
    }
  }
  let replacements : Array[
    (Int32, PreparedPagedGraphStep, PreparedPagedGraphStep),
  ] = []
  for item in prepared {
    replacements.push((item.operation_id.as_int(), item.partial, item.merge))
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
fn append_paged_sampling_suffix(
  model_steps : ReadOnlyArray[PreparedPagedGraphStep],
  complete_steps : ReadOnlyArray[PreparedPagedGraphStep],
  decode_model_steps : ReadOnlyArray[PreparedPagedGraphStep],
) -> ReadOnlyArray[PreparedPagedGraphStep] raise DeviceStepError {
  if complete_steps.length() < model_steps.length() {
    raise InvalidExecutor(ResourceState)
  }
  let result : Array[PreparedPagedGraphStep] = []
  for step in decode_model_steps {
    result.push(step)
  }
  for index in model_steps.length()..<complete_steps.length() {
    result.push(complete_steps[index])
  }
  ReadOnlyArray::from_array(result)
}
