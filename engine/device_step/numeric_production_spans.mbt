///|
fn replace_paged_production_spans(
  baseline : ReadOnlyArray[PreparedPagedGraphStep],
  residual_replacement : PreparedPagedGraphStep?,
  residual_span : ApprovedFusedResidualRmsNormSpan?,
  qkv_readonly_replacements : (PreparedPagedGraphStep, PreparedPagedGraphStep)?,
  qkv_readonly_span : ApprovedFusedQkvReadOnlyAttentionSpan?,
) -> ReadOnlyArray[PreparedPagedGraphStep] raise DeviceStepError {
  let residual_indices = match (residual_replacement, residual_span) {
    (Some(_), Some(span)) => Some(span.operation_indices())
    (None, None) => None
    _ => raise InvalidExecutor(ResourceState)
  }
  let qkv_indices = match (qkv_readonly_replacements, qkv_readonly_span) {
    (Some(_), Some(span)) => Some(span.operation_indices())
    (None, None) => None
    _ => raise InvalidExecutor(ResourceState)
  }
  match (residual_indices, qkv_indices) {
    (Some(residual), Some(qkv)) =>
      if residual.0 == qkv.0 ||
        residual.0 == qkv.1 ||
        residual.0 == qkv.2 ||
        residual.1 == qkv.0 ||
        residual.1 == qkv.1 ||
        residual.1 == qkv.2 {
        raise InvalidExecutor(ResourceState)
      }
    _ => ()
  }
  match (residual_indices, qkv_indices, qkv_readonly_replacements) {
    (None, Some(indices), Some(replacements)) =>
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
    match (residual_indices, residual_replacement) {
      (Some(indices), Some(replacement)) => {
        if index == indices.0 {
          output.push(replacement)
          continue
        }
        if index == indices.1 {
          continue
        }
      }
      _ => ()
    }
    output.push(step)
  }
  ReadOnlyArray::from_array(output)
}
