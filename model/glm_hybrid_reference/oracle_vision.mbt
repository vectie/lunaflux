///|
fn oracle_dense_vision_attention(
  query : ArrayView[Double],
  key : ArrayView[Double],
  value : ArrayView[Double],
  patch_count : Int,
  head_count : Int,
  head_dimension : Int,
  limits : GlmTinyOracleLimits,
) -> Array[Double] raise GlmHybridReferenceError {
  let width = head_count * head_dimension
  if query.length() != patch_count * width ||
    key.length() != query.length() ||
    value.length() != query.length() {
    raise InvalidVisionGeometry
  }
  oracle_require_finite(query, limits)
  oracle_require_finite(key, limits)
  oracle_require_finite(value, limits)
  let output = Array::make(query.length(), 0.0)
  let scores = Array::make(patch_count, 0.0)
  let scale = 1.0 / head_dimension.to_double().sqrt()
  for patch in 0..<patch_count {
    for head in 0..<head_count {
      let query_base = patch * width + head * head_dimension
      let mut maximum = -@double.infinity
      for source in 0..<patch_count {
        let source_base = source * width + head * head_dimension
        let mut dot = 0.0
        for component in 0..<head_dimension {
          dot += query[query_base + component] * key[source_base + component]
        }
        scores[source] = dot * scale
        if scores[source] > maximum {
          maximum = scores[source]
        }
      }
      let mut denominator = 0.0
      for source in 0..<patch_count {
        scores[source] = @math.exp(scores[source] - maximum)
        denominator += scores[source]
      }
      if !finite(denominator) || denominator <= 0.0 {
        raise InvalidProbability
      }
      for source in 0..<patch_count {
        let probability = scores[source] / denominator
        let source_base = source * width + head * head_dimension
        for component in 0..<head_dimension {
          output[query_base + component] += probability *
            value[source_base + component]
        }
      }
    }
  }
  oracle_require_finite(output, limits)
  output
}

///|
fn oracle_vision_merge(
  input : ArrayView[Double],
  weights : ArrayView[Double],
  bias : ArrayView[Double],
  patch_count : Int,
  width : Int,
  merge_size : Int,
  limits : GlmTinyOracleLimits,
) -> Array[Double] raise GlmHybridReferenceError {
  let area = merge_size * merge_size
  let row_count = patch_count / area
  if input.length() != patch_count * width ||
    weights.length() != width * width * area ||
    bias.length() != width {
    raise InvalidVisionGeometry
  }
  oracle_require_finite(input, limits)
  oracle_require_finite(weights, limits)
  oracle_require_finite(bias, limits)
  let output = Array::make(row_count * width, 0.0)
  for row in 0..<row_count {
    for destination in 0..<width {
      let mut sum = bias[destination]
      for source in 0..<width {
        for spatial in 0..<area {
          sum += input[(row * area + spatial) * width + source] *
            weights[(destination * width + source) * area + spatial]
        }
      }
      if !finite(sum) {
        raise NonFiniteInput
      }
      output[row * width + destination] = sum
    }
  }
  output
}

///|
/// Executes one bounded bidirectional vision attention, SwiGLU MLP, spatial
/// merge, and projector block. It is a correctness oracle only.
pub fn execute_tiny_vision_block(
  patches : ArrayView[Double],
  plan : GlmTinyVisionBlockPlan,
  weights : GlmTinyVisionBlockWeights,
  limits~ : GlmTinyOracleLimits,
) -> GlmVisionReferenceResult raise GlmHybridReferenceError {
  let width = plan.head_count * plan.head_dimension
  if patches.length() != plan.patch_count * width ||
    weights.attention_norm.length() != width ||
    weights.query.length() != width * width ||
    weights.key.length() != width * width ||
    weights.value.length() != width * width ||
    weights.attention_output.length() != width * width ||
    weights.ffn_norm.length() != width ||
    weights.gate.length() != plan.intermediate_size * width ||
    weights.up.length() != plan.intermediate_size * width ||
    weights.down.length() != width * plan.intermediate_size ||
    weights.project.length() != plan.project_width * width ||
    weights.project_bias.length() != plan.project_width {
    raise InvalidVisionGeometry
  }
  let normalized = oracle_rms_norm(
    patches,
    weights.attention_norm,
    plan.patch_count,
    width,
    plan.rms_epsilon,
    limits,
  )
  let query = oracle_linear(
    normalized,
    weights.query,
    plan.patch_count,
    width,
    width,
    limits,
  )
  let key = oracle_linear(
    normalized,
    weights.key,
    plan.patch_count,
    width,
    width,
    limits,
  )
  let value = oracle_linear(
    normalized,
    weights.value,
    plan.patch_count,
    width,
    width,
    limits,
  )
  let attended = oracle_dense_vision_attention(
    query,
    key,
    value,
    plan.patch_count,
    plan.head_count,
    plan.head_dimension,
    limits,
  )
  let projected_attention = oracle_linear(
    attended,
    weights.attention_output,
    plan.patch_count,
    width,
    width,
    limits,
  )
  let attention_state = oracle_add(patches, projected_attention, limits)
  let ffn_input = oracle_rms_norm(
    attention_state,
    weights.ffn_norm,
    plan.patch_count,
    width,
    plan.rms_epsilon,
    limits,
  )
  let ffn = oracle_swiglu(
    ffn_input,
    weights.gate,
    weights.up,
    weights.down,
    plan.patch_count,
    width,
    plan.intermediate_size,
    limits,
  )
  let block_output = oracle_add(attention_state, ffn, limits)
  let merged = oracle_vision_merge(
    block_output,
    weights.merge,
    weights.merge_bias,
    plan.patch_count,
    width,
    plan.merge_size,
    limits,
  )
  let row_count = plan.patch_count / (plan.merge_size * plan.merge_size)
  let projected = oracle_linear(
    merged,
    weights.project,
    row_count,
    width,
    plan.project_width,
    limits,
  )
  let output = Array::make(projected.length(), 0.0)
  for row in 0..<row_count {
    for column in 0..<plan.project_width {
      let value = projected[row * plan.project_width + column] +
        weights.project_bias[column]
      if !finite(value) {
        raise NonFiniteInput
      }
      output[row * plan.project_width + column] = value
    }
  }
  {
    row_count,
    width: plan.project_width,
    values: ReadOnlyArray::from_array(output),
  }
}
