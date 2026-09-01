///|
/// Replaces one or more four-operation projection -> normalization ->
/// position transform -> attention spans with a three-launch lowering:
///
///   original projection -> fused postprocess -> read-only attention
///
/// The projection is deliberately retained. Semantic fusion may lower to more
/// than one backend launch when that preserves a faster shape-selected matrix
/// implementation. All indices and replacements are prepared during startup;
/// the returned array becomes the immutable execution order.
fn[Step] replace_projection_postprocess_attention_spans(
  baseline : ReadOnlyArray[Step],
  projection_indices : ReadOnlyArray[Int],
  postprocess_steps : ReadOnlyArray[Step],
  readonly_attention_steps : ReadOnlyArray[Step],
) -> ReadOnlyArray[Step] raise DeviceStepError {
  if projection_indices.length() != postprocess_steps.length() ||
    projection_indices.length() != readonly_attention_steps.length() {
    raise InvalidExecutor(ResourceState)
  }
  let replacement_for = FixedArray::make(baseline.length(), -1)
  let skipped = FixedArray::make(baseline.length(), false)
  for replacement_index, projection_index in projection_indices {
    if projection_index < 0 ||
      projection_index > baseline.length() - 4 ||
      replacement_for[projection_index + 1] >= 0 {
      raise InvalidExecutor(ResourceState)
    }
    for offset in 0..<4 {
      if skipped[projection_index + offset] {
        raise InvalidExecutor(ResourceState)
      }
      skipped[projection_index + offset] = true
    }
    // Keep the existing projection launch and replace the following three
    // semantic operations at their first position.
    skipped[projection_index] = false
    replacement_for[projection_index + 1] = replacement_index
  }
  let output : Array[Step] = []
  for index, step in baseline {
    let replacement = replacement_for[index]
    if replacement >= 0 {
      output.push(postprocess_steps[replacement])
      output.push(readonly_attention_steps[replacement])
    } else if !skipped[index] {
      output.push(step)
    }
  }
  ReadOnlyArray::from_array(output)
}
