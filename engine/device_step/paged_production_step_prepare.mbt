///|
/// Prepares optional production replacements and publishes one complete step
/// array. Any resource or argument failure occurs before the ordered executor
/// is created, so the caller can roll back the whole preparation atomically.
fn prepare_paged_production_steps(
  context : @device.Context,
  blueprint : PagedKvExecutionBlueprint,
  function_indices : FixedArray[Int],
  split_decode_indices : FixedArray[Int],
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation : @device.Allocation,
  kv : @device.Allocation,
  resources : PagedGraphResources,
  fused_residual_span : ApprovedFusedResidualRmsNormSpan?,
  fused_qkv_readonly_span : ApprovedFusedQkvReadOnlyAttentionSpan?,
) -> ReadOnlyArray[PreparedPagedGraphStep] raise DeviceStepError {
  let mut fused_residual_function : Int? = None
  match fused_residual_span {
    Some(span) =>
      fused_residual_function = Some(
        prepare_fused_residual_resources(context, span, resources),
      )
    None => ()
  }
  let mut fused_qkv_readonly_functions : (Int, Int)? = None
  match fused_qkv_readonly_span {
    Some(span) =>
      fused_qkv_readonly_functions = Some(
        prepare_fused_qkv_readonly_resources(context, span, resources),
      )
    None => ()
  }
  let baseline = prepare_paged_steps(
    blueprint, function_indices, descriptor, weight, activation, kv,
  )
  let residual_replacement = match
    (fused_residual_span, fused_residual_function) {
    (Some(span), Some(function_index)) =>
      Some(
        prepare_fused_residual_step(
          blueprint, span, function_index, resources, descriptor, weight, activation,
          kv,
        ),
      )
    (None, None) => None
    _ => raise InvalidExecutor(ResourceState)
  }
  let qkv_readonly_replacements = match
    (fused_qkv_readonly_span, fused_qkv_readonly_functions) {
    (Some(span), Some(functions)) =>
      Some(
        prepare_fused_qkv_readonly_steps(
          blueprint,
          span,
          functions.0,
          functions.1,
          descriptor,
          weight,
          activation,
          kv,
        ),
      )
    (None, None) => None
    _ => raise InvalidExecutor(ResourceState)
  }
  let replaced = replace_paged_production_spans(
    baseline, residual_replacement, fused_residual_span, qkv_readonly_replacements,
    fused_qkv_readonly_span,
  )
  let expanded : Array[PreparedPagedGraphStep] = []
  for prepared in replaced {
    expanded.push(prepared)
    let mut split_function_index = -1
    for step_index, step in blueprint.steps() {
      if step.operation_id() == prepared.operation_id {
        split_function_index = split_decode_indices[step_index]
        break
      }
    }
    if split_function_index >= 0 {
      expanded.push({
        operation_id: prepared.operation_id,
        function_index: split_function_index,
        dimensions: prepared.dimensions,
        arguments: prepared.arguments,
      })
    }
  }
  ReadOnlyArray::from_array(expanded)
}
