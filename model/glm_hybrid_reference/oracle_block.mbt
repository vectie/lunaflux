///|
fn oracle_block_input(
  input : ArrayView[Double],
  plan : GlmTinyHybridBlockPlan,
  width : Int,
  limits : GlmTinyOracleLimits,
) -> Array[Double] raise GlmHybridReferenceError {
  let feature_count = plan.token_count * width
  match plan.composition {
    Residual => {
      if input.length() != feature_count {
        raise InvalidShape
      }
      oracle_require_finite(input, limits)
      input.to_owned()
    }
    MultiStreamHyperConnection(stream_count~, branch_stream~, ..) => {
      if input.length() != stream_count * feature_count {
        raise InvalidShape
      }
      oracle_require_finite(input, limits)
      input[branch_stream * feature_count:(branch_stream + 1) * feature_count].to_owned()
    }
  }
}

///|
fn oracle_compose_output(
  original : ArrayView[Double],
  branch_output : ArrayView[Double],
  mixing : ArrayView[Double],
  plan : GlmTinyHybridBlockPlan,
  width : Int,
  limits : GlmTinyOracleLimits,
) -> Array[Double] raise GlmHybridReferenceError {
  let feature_count = plan.token_count * width
  match plan.composition {
    Residual => {
      if !mixing.is_empty() {
        raise InvalidShape
      }
      branch_output.to_owned()
    }
    MultiStreamHyperConnection(stream_count~, branch_stream~, output_stream~) => {
      if original.length() != stream_count * feature_count ||
        mixing.length() != stream_count * stream_count {
        raise InvalidShape
      }
      oracle_require_finite(mixing, limits)
      let streams = original.to_owned()
      for feature in 0..<feature_count {
        streams[branch_stream * feature_count + feature] = branch_output[feature]
      }
      let output = Array::make(feature_count, 0.0)
      for source in 0..<stream_count {
        let weight = mixing[output_stream * stream_count + source]
        for feature in 0..<feature_count {
          output[feature] += weight * streams[source * feature_count + feature]
        }
      }
      oracle_require_finite(output, limits)
      output
    }
  }
}

///|
/// Executes one bounded GLM-style hybrid decoder block as a scalar oracle.
/// This correctness-only API is not a production fallback or performance path.
pub fn execute_tiny_hybrid_block(
  input : ArrayView[Double],
  selected_indices : ArrayView[Int],
  plan : GlmTinyHybridBlockPlan,
  weights : GlmTinyHybridBlockWeights,
  limits~ : GlmTinyOracleLimits,
) -> GlmTinyHybridBlockResult raise GlmHybridReferenceError {
  let width = plan.head_count * plan.head_dimension
  let branch_input = oracle_block_input(input, plan, width, limits)
  if weights.attention_norm.length() != width ||
    weights.query_a.length() != plan.low_rank * width ||
    weights.query_b.length() != width * plan.low_rank ||
    weights.key_a.length() != plan.low_rank * width ||
    weights.key_b.length() != width * plan.low_rank ||
    weights.value_a.length() != plan.low_rank * width ||
    weights.value_b.length() != width * plan.low_rank ||
    weights.attention_output.length() != width * width ||
    weights.ffn_norm.length() != width {
    raise InvalidShape
  }
  let normalized = oracle_rms_norm(
    branch_input,
    weights.attention_norm,
    plan.token_count,
    width,
    plan.rms_epsilon,
    limits,
  )
  let query = oracle_low_rank_projection(
    normalized,
    weights.query_a,
    weights.query_b,
    plan.token_count,
    width,
    plan.low_rank,
    limits,
  )
  let key = oracle_low_rank_projection(
    normalized,
    weights.key_a,
    weights.key_b,
    plan.token_count,
    width,
    plan.low_rank,
    limits,
  )
  let value = oracle_low_rank_projection(
    normalized,
    weights.value_a,
    weights.value_b,
    plan.token_count,
    width,
    plan.low_rank,
    limits,
  )
  let sparse = oracle_sparse_attention(
    query,
    key,
    value,
    selected_indices,
    plan.token_count,
    plan.head_count,
    plan.head_dimension,
    plan.sparse_top_k,
    limits,
  )
  let attention_projection = oracle_linear(
    sparse,
    weights.attention_output,
    plan.token_count,
    width,
    width,
    limits,
  )
  let attention_state = oracle_add(branch_input, attention_projection, limits)
  let ffn_input = oracle_rms_norm(
    attention_state,
    weights.ffn_norm,
    plan.token_count,
    width,
    plan.rms_epsilon,
    limits,
  )
  let (ffn_branch, selected_experts) = oracle_feed_forward(
    ffn_input,
    plan,
    weights.feed_forward,
    width,
    limits,
  )
  let branch_output = oracle_add(attention_state, ffn_branch, limits)
  let output = oracle_compose_output(
    input,
    branch_output,
    weights.hyper_mixing,
    plan,
    width,
    limits,
  )
  {
    attention_branch: ReadOnlyArray::from_array(attention_projection),
    feed_forward_branch: ReadOnlyArray::from_array(ffn_branch),
    output: ReadOnlyArray::from_array(output),
    selected_experts: ReadOnlyArray::from_array(selected_experts),
  }
}
