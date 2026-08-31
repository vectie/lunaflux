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
  fused_residual_spans : ReadOnlyArray[ApprovedFusedResidualRmsNormSpan],
  fused_qkv_readonly_span : ApprovedFusedQkvReadOnlyAttentionSpan?,
  qwen_fused_attention_spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
) -> ReadOnlyArray[PreparedPagedGraphStep] raise DeviceStepError {
  let fused_residual_function = match fused_residual_spans.get(0) {
    Some(span) =>
      Some(prepare_fused_residual_resources(context, span, resources))
    None => None
  }
  let mut fused_qkv_readonly_functions : (Int, Int)? = None
  match fused_qkv_readonly_span {
    Some(span) =>
      fused_qkv_readonly_functions = Some(
        prepare_fused_qkv_readonly_resources(context, span, resources),
      )
    None => ()
  }
  let qwen_fused_attention_functions = if qwen_fused_attention_spans.is_empty() {
    None
  } else {
    Some(
      prepare_qwen_fused_attention_resources(
        context, qwen_fused_attention_spans, resources,
      ),
    )
  }
  let baseline = prepare_paged_steps(
    blueprint, function_indices, descriptor, weight, activation, kv,
  )
  let residual_replacements : Array[PreparedPagedGraphStep] = []
  match fused_residual_function {
    Some(function_index) =>
      for span in fused_residual_spans {
        residual_replacements.push(
          prepare_fused_residual_step(
            blueprint, span, function_index, resources, descriptor, weight, activation,
            kv,
          ),
        )
      }
    None =>
      if !fused_residual_spans.is_empty() {
        raise InvalidExecutor(ResourceState)
      }
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
  let qwen_ingress_replacements : Array[PreparedPagedGraphStep] = []
  let qwen_readonly_replacements : Array[PreparedPagedGraphStep] = []
  match qwen_fused_attention_functions {
    Some(functions) =>
      for span in qwen_fused_attention_spans {
        let prepared = prepare_qwen_fused_attention_steps(
          blueprint,
          span,
          functions.0,
          functions.1,
          descriptor,
          weight,
          activation,
          kv,
        )
        qwen_ingress_replacements.push(prepared.0)
        qwen_readonly_replacements.push(prepared.1)
      }
    None =>
      if !qwen_fused_attention_spans.is_empty() {
        raise InvalidExecutor(ResourceState)
      }
  }
  let replaced = replace_paged_production_spans(
    baseline,
    ReadOnlyArray::from_array(residual_replacements),
    fused_residual_spans,
    qkv_readonly_replacements,
    fused_qkv_readonly_span,
    ReadOnlyArray::from_array(qwen_ingress_replacements),
    ReadOnlyArray::from_array(qwen_readonly_replacements),
    qwen_fused_attention_spans,
  )
  let expanded : Array[PreparedPagedGraphStep] = []
  for prepared in replaced {
    expanded.push(prepared)
    let split_function_index = paged_production_split_decode_function(
      prepared.operation_id,
      blueprint,
      split_decode_indices,
      qwen_fused_attention_spans,
    )
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

///|
/// A Qwen read-only replacement already covers mixed prefill/decode attention
/// and retains the semantic attention operation id. It must not inherit a
/// legacy split-decode companion attached to the eliminated baseline step.
fn paged_production_split_decode_function(
  operation : @plan.OperationId,
  blueprint : PagedKvExecutionBlueprint,
  split_decode_indices : FixedArray[Int],
  qwen_spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
) -> Int {
  for span in qwen_spans {
    let indices = span.operation_indices()
    match blueprint.steps().get(indices.3) {
      Some(attention) => if attention.operation_id() == operation { return -1 }
      None => ()
    }
  }
  for step_index, step in blueprint.steps() {
    if step.operation_id() == operation {
      return split_decode_indices[step_index]
    }
  }
  -1
}
