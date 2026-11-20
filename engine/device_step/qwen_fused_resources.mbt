///|
fn qwen_fused_postprocess_argument_count() -> Int32 {
  12
}

///|
fn qwen_fused_full_ingress_argument_count() -> Int32 {
  15
}

///|
priv struct PreparedQwenFusedAttentionResources {
  ingress_function : Int32
  rotary_prepare_function : Int32?
  prefill_function : Int32
  wide_prefill_function : Int32?
  decode_function : Int32
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
  let rotated_shape = qwen_fused_attention_sidecar_shape(blueprint, first)
  let sidecar_shape = qwen_step_rotary_sidecar_shape(
    rotated_shape,
    first.step_rotary_cache,
  )
  for span in spans {
    if !span.same_runtime_modules(first) ||
      qwen_fused_attention_sidecar_shape(blueprint, span) != rotated_shape {
      raise InvalidExecutor(ResourceState)
    }
  }
  let postprocess = prepare_composite_function(
    context,
    first.postprocess_module_bytes(),
    first.postprocess_function_symbol(),
    resources,
  )
  if first.ingress_row_variant is Some(variant) {
    let function_index = prepare_composite_function(
      context,
      variant.module_bytes(),
      variant.function_symbol(),
      resources,
    )
    resources.projection_variants.push({
      base_function_index: postprocess,
      maximum_token_rows: variant.maximum_token_rows(),
      function_index,
      launch: QueryTileCappedGridX(
        variant.launch_dimensions(),
        variant.query_tile_rows(),
      ),
    })
  }
  let readonly_function = prepare_composite_function(
    context,
    first.readonly_module_bytes(),
    first.readonly_function_symbol(),
    resources,
  )
  let rotary_prepare_function = match first.step_rotary_cache {
    Some(_) =>
      Some(
        prepare_composite_function(
          context,
          first.postprocess_module_bytes(),
          first.postprocess_function_symbol() + "_rotary_prepare",
          resources,
        ),
      )
    None => None
  }
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
    rotary_prepare_function,
    prefill_function: readonly_function,
    wide_prefill_function,
    decode_function,
    sidecar,
  }
}

///|
fn qwen_fused_attention_weights(
  step : PagedKvExecutionStep,
  expected : Int32,
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
