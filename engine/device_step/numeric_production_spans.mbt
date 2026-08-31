///|
fn replace_paged_production_spans(
  baseline : ReadOnlyArray[PreparedPagedGraphStep],
  residual_replacements : ReadOnlyArray[PreparedPagedGraphStep],
  residual_spans : ReadOnlyArray[ApprovedFusedResidualRmsNormSpan],
  qkv_readonly_replacements : (PreparedPagedGraphStep, PreparedPagedGraphStep)?,
  qkv_readonly_span : ApprovedFusedQkvReadOnlyAttentionSpan?,
) -> ReadOnlyArray[PreparedPagedGraphStep] raise DeviceStepError {
  if residual_replacements.length() != residual_spans.length() {
    raise InvalidExecutor(ResourceState)
  }
  let qkv_indices = match (qkv_readonly_replacements, qkv_readonly_span) {
    (Some(_), Some(span)) => Some(span.operation_indices())
    (None, None) => None
    _ => raise InvalidExecutor(ResourceState)
  }
  for span in residual_spans {
    let residual = span.operation_indices()
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
  match (residual_spans.is_empty(), qkv_indices, qkv_readonly_replacements) {
    (true, Some(indices), Some(replacements)) =>
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
