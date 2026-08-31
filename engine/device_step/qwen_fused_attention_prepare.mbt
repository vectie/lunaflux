///|
fn qwen_fused_ingress_argument_count() -> Int {
  15
}

///|
fn prepare_qwen_fused_attention_resources(
  context : @device.Context,
  spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
  resources : PagedGraphResources,
) -> (Int, Int) raise DeviceStepError {
  guard spans.get(0) is Some(first) else {
    raise InvalidExecutor(ResourceState)
  }
  for span in spans {
    if !span.same_runtime_modules(first) {
      raise InvalidExecutor(ResourceState)
    }
  }
  let ingress = prepare_composite_function(
    context,
    first.ingress_module_bytes(),
    first.ingress_function_symbol(),
    resources,
  )
  let readonly_function = prepare_composite_function(
    context,
    first.readonly_module_bytes(),
    first.readonly_function_symbol(),
    resources,
  )
  (ingress, readonly_function)
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
fn qwen_fused_regions_overlap(
  left : PagedKvOperandBinding,
  right : PagedKvOperandBinding,
) -> Bool {
  if left.space() != right.space() {
    return false
  }
  left.offset_bytes() < right.offset_bytes() + right.byte_count() &&
  right.offset_bytes() < left.offset_bytes() + left.byte_count()
}

///|
fn prepare_qwen_fused_attention_steps(
  blueprint : PagedKvExecutionBlueprint,
  span : ApprovedQwenFusedAttentionSpan,
  ingress_function : Int,
  readonly_function : Int,
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation : @device.Allocation,
  kv : @device.Allocation,
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
  let qkv_input = unique_composite_binding(qkv_step, role => {
    role is ActivationInput(_)
  })
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
  let qkv_weights = qwen_fused_attention_weights(qkv_step, 3)
  let qk_norm_weights = qwen_fused_attention_weights(qk_norm_step, 2)
  let output = unique_composite_binding(attention_step, role => {
    role is ActivationOutput(_)
  })
  let key_cache = unique_composite_binding(attention_step, role => {
    role is PersistentKvComponent(_, Key)
  })
  let value_cache = unique_composite_binding(attention_step, role => {
    role is PersistentKvComponent(_, Value)
  })
  if qwen_fused_regions_overlap(qkv_input, rotated) ||
    qwen_fused_regions_overlap(rotated, output) ||
    qwen_fused_regions_overlap(rotated, key_cache) ||
    qwen_fused_regions_overlap(rotated, value_cache) {
    raise InvalidExecutor(ResourceState)
  }
  for binding in qkv_weights {
    if qwen_fused_regions_overlap(rotated, binding) {
      raise InvalidExecutor(ResourceState)
    }
  }
  for binding in qk_norm_weights {
    if qwen_fused_regions_overlap(rotated, binding) {
      raise InvalidExecutor(ResourceState)
    }
  }
  let ingress_bindings : Array[PagedKvOperandBinding] = []
  for binding in runtime {
    ingress_bindings.push(binding)
  }
  ingress_bindings.push(qkv_input)
  for binding in qkv_weights {
    ingress_bindings.push(binding)
  }
  for binding in qk_norm_weights {
    ingress_bindings.push(binding)
  }
  for binding in [rotated, key_cache, value_cache] {
    ingress_bindings.push(binding)
  }
  let readonly_bindings : Array[PagedKvOperandBinding] = []
  for binding in runtime {
    readonly_bindings.push(binding)
  }
  for binding in [readonly_input, output, key_cache, value_cache] {
    readonly_bindings.push(binding)
  }
  if ingress_bindings.length() != qwen_fused_ingress_argument_count() ||
    readonly_bindings.length() != readonly_attention_production_argument_count() {
    raise InvalidExecutor(ResourceState)
  }
  (
    prepare_composite_launch(
      qkv_step.operation_id(),
      ingress_function,
      span.ingress_launch(),
      ingress_bindings,
      descriptor,
      weight,
      activation,
      kv,
    ),
    prepare_composite_launch(
      attention_step.operation_id(),
      readonly_function,
      span.readonly_launch(),
      readonly_bindings,
      descriptor,
      weight,
      activation,
      kv,
    ),
  )
}

///|
fn[Step] replace_qwen_fused_attention_spans(
  baseline : ReadOnlyArray[Step],
  qkv_indices : ReadOnlyArray[Int],
  ingress : ReadOnlyArray[Step],
  readonly_steps : ReadOnlyArray[Step],
) -> ReadOnlyArray[Step] raise DeviceStepError {
  if qkv_indices.length() != ingress.length() ||
    qkv_indices.length() != readonly_steps.length() {
    raise InvalidExecutor(ResourceState)
  }
  let replacement_for = FixedArray::make(baseline.length(), -1)
  let skipped = FixedArray::make(baseline.length(), false)
  for replacement_index, qkv_index in qkv_indices {
    if qkv_index < 0 ||
      qkv_index > baseline.length() - 4 ||
      replacement_for[qkv_index] >= 0 {
      raise InvalidExecutor(ResourceState)
    }
    for offset in 0..<4 {
      if skipped[qkv_index + offset] {
        raise InvalidExecutor(ResourceState)
      }
      skipped[qkv_index + offset] = true
    }
    replacement_for[qkv_index] = replacement_index
  }
  let output : Array[Step] = []
  for index, step in baseline {
    let replacement = replacement_for[index]
    if replacement >= 0 {
      output.push(ingress[replacement])
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
