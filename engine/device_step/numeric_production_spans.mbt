///|
fn replace_paged_production_spans(
  baseline : ReadOnlyArray[PreparedPagedGraphStep],
  residual_replacements : ReadOnlyArray[PreparedPagedGraphStep],
  residual_spans : ReadOnlyArray[ApprovedFusedResidualRmsNormSpan],
  qkv_readonly_replacements : (PreparedPagedGraphStep, PreparedPagedGraphStep)?,
  qkv_readonly_span : ApprovedFusedQkvReadOnlyAttentionSpan?,
  qwen_ingress_replacements : ReadOnlyArray[PreparedPagedGraphStep],
  qwen_readonly_replacements : ReadOnlyArray[PreparedPagedGraphStep],
  qwen_spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
) -> ReadOnlyArray[PreparedPagedGraphStep] raise DeviceStepError {
  if residual_replacements.length() != residual_spans.length() {
    raise InvalidExecutor(ResourceState)
  }
  let qkv_indices = match (qkv_readonly_replacements, qkv_readonly_span) {
    (Some(_), Some(span)) => Some(span.operation_indices())
    (None, None) => None
    _ => raise InvalidExecutor(ResourceState)
  }
  if qwen_ingress_replacements.length() != qwen_spans.length() ||
    qwen_readonly_replacements.length() != qwen_spans.length() {
    raise InvalidExecutor(ResourceState)
  }
  let qwen_starts = FixedArray::make(baseline.length(), -1)
  let qwen_covered = FixedArray::make(baseline.length(), false)
  for span_index, span in qwen_spans {
    let indices = span.operation_indices()
    if indices.0 < 0 ||
      indices.1 != indices.0 + 1 ||
      indices.2 != indices.0 + 2 ||
      indices.3 != indices.0 + 3 ||
      indices.3 >= baseline.length() ||
      qwen_starts[indices.0] >= 0 {
      raise InvalidExecutor(ResourceState)
    }
    for index in [indices.0, indices.1, indices.2, indices.3] {
      if qwen_covered[index] {
        raise InvalidExecutor(ResourceState)
      }
      qwen_covered[index] = true
    }
    qwen_starts[indices.0] = span_index
  }
  for span in residual_spans {
    let residual = span.operation_indices()
    if qwen_fused_attention_overlaps_residual(
        qwen_spans,
        residual.0,
        residual.1,
      ) {
      raise InvalidExecutor(ResourceState)
    }
    match qkv_indices {
      Some(qkv) =>
        if residual.0 == qkv.0 ||
          residual.0 == qkv.1 ||
          residual.0 == qkv.2 ||
          residual.1 == qkv.0 ||
          residual.1 == qkv.1 ||
          residual.1 == qkv.2 {
          raise InvalidExecutor(ResourceState)
        }
      None => ()
    }
  }
  match qkv_indices {
    Some(indices) =>
      if qwen_covered[indices.0] ||
        qwen_covered[indices.1] ||
        qwen_covered[indices.2] {
        raise InvalidExecutor(ResourceState)
      }
    None => ()
  }
  if residual_spans.is_empty() && qkv_indices is None && !qwen_spans.is_empty() {
    let qwen_indices : Array[Int] = []
    for span in qwen_spans {
      qwen_indices.push(span.operation_indices().0)
    }
    return replace_qwen_fused_attention_spans(
      baseline,
      ReadOnlyArray::from_array(qwen_indices),
      qwen_ingress_replacements,
      qwen_readonly_replacements,
    )
  }
  match
    (
      residual_spans.is_empty(),
      qwen_spans.is_empty(),
      qkv_indices,
      qkv_readonly_replacements,
    ) {
    (true, true, Some(indices), Some(replacements)) =>
      return replace_fused_qkv_readonly_span(
        baseline,
        replacements.0,
        replacements.1,
        indices.0,
        indices.1,
        indices.2,
      )
    _ => ()
  }
  let output : Array[PreparedPagedGraphStep] = []
  for index, step in baseline {
    let qwen_replacement = qwen_starts[index]
    if qwen_replacement >= 0 {
      output.push(qwen_ingress_replacements[qwen_replacement])
      output.push(qwen_readonly_replacements[qwen_replacement])
      continue
    }
    if qwen_covered[index] {
      continue
    }
    match (qkv_indices, qkv_readonly_replacements) {
      (Some(indices), Some(replacements)) => {
        if index == indices.0 {
          output.push(replacements.0)
          output.push(replacements.1)
          continue
        }
        if index == indices.1 || index == indices.2 {
          continue
        }
      }
      _ => ()
    }
    let mut residual_replaced = false
    for span_index, span in residual_spans {
      let indices = span.operation_indices()
      if index == indices.0 {
        output.push(residual_replacements[span_index])
        residual_replaced = true
        break
      }
      if index == indices.1 {
        residual_replaced = true
        break
      }
    }
    if residual_replaced {
      continue
    }
    output.push(step)
  }
  ReadOnlyArray::from_array(output)
}
