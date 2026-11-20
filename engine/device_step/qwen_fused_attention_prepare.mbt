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
  function_index : Int32,
  launch : @launch_contract.AotLaunchDimensions,
  bindings : ArrayView[PagedKvOperandBinding],
  sidecar_binding_index : Int32,
  sidecar : @device.Allocation,
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation : @device.Allocation,
  kv : @device.Allocation,
  prefill_only~ : Bool = false,
  metadata? : @device.KernelArgument,
  rotary_cache? : @device.KernelArgument,
) -> PreparedPagedGraphStep raise DeviceStepError {
  if sidecar_binding_index < 0 || sidecar_binding_index >= bindings.length() {
    raise executor_operation_error(operation, ResourceState)
  }
  let arguments : Array[@device.KernelArgument] = []
  for index, binding in bindings {
    arguments.push(
      if binding.role() == RuntimeInput(QueryRowOffsets) &&
        metadata is Some(argument) {
        argument
      } else if prefill_only && binding.role() == RuntimeInput(StepCounts) {
        prefill_attention_counts_argument(descriptor)
      } else if index == sidecar_binding_index {
        qwen_fused_sidecar_argument(binding, sidecar)
      } else {
        composite_binding_argument(binding, descriptor, weight, activation, kv)
      },
    )
  }
  if rotary_cache is Some(argument) {
    arguments.push(argument)
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
  ingress_function : Int32,
  readonly_function : Int32,
  readonly_launch : @launch_contract.AotLaunchDimensions,
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation : @device.Allocation,
  kv : @device.Allocation,
  sidecar : @device.Allocation,
  prefill_only~ : Bool = false,
  query_metadata~ : Bool = false,
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
      rotary_cache?=match span.step_rotary_cache {
        Some(cache) => Some(qwen_step_rotary_argument(sidecar, cache))
        None => None
      },
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
      prefill_only~,
      metadata?=if query_metadata {
        Some(attention_metadata_argument(descriptor, blueprint.profile()))
      } else {
        None
      },
    ),
  )
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
    let qkv_indices : Array[Int32] = []
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
  residual_index : Int32,
  norm_index : Int32,
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
