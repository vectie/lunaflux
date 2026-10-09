///|
/// Hard limits for tiny end-to-end correctness oracles. These deliberately
/// cannot represent production checkpoint geometry.
pub struct GlmTinyOracleLimits {
  priv maximum_tokens : Int
  priv maximum_width : Int
  priv maximum_low_rank : Int
  priv maximum_intermediate : Int
  priv maximum_experts : Int
  priv maximum_selected : Int
  priv maximum_values : Int
}

///|
pub fn GlmTinyOracleLimits::new(
  maximum_tokens~ : Int,
  maximum_width~ : Int,
  maximum_low_rank~ : Int,
  maximum_intermediate~ : Int,
  maximum_experts~ : Int,
  maximum_selected~ : Int,
  maximum_values~ : Int,
) -> GlmTinyOracleLimits raise GlmHybridReferenceError {
  if maximum_tokens <= 0 ||
    maximum_tokens > 32 ||
    maximum_width <= 0 ||
    maximum_width > 128 ||
    maximum_low_rank <= 0 ||
    maximum_low_rank > 128 ||
    maximum_intermediate <= 0 ||
    maximum_intermediate > 512 ||
    maximum_experts <= 0 ||
    maximum_experts > 64 ||
    maximum_selected <= 0 ||
    maximum_selected > 32 ||
    maximum_values <= 0 ||
    maximum_values > 1048576 {
    raise InvalidLimits
  }
  {
    maximum_tokens,
    maximum_width,
    maximum_low_rank,
    maximum_intermediate,
    maximum_experts,
    maximum_selected,
    maximum_values,
  }
}

///|
pub(all) enum GlmTinyFeedForwardPlan {
  Dense
  Moe(
    expert_count~ : Int,
    experts_per_token~ : Int,
    score_function~ : @advanced_decoder_execution_plan.ExpertScoreFunction,
    normalize_selected~ : Bool,
    route_scale~ : Float
  )
} derive(Debug, Eq)

///|
pub(all) enum GlmTinyCompositionPlan {
  Residual
  MultiStreamHyperConnection(
    stream_count~ : Int,
    branch_stream~ : Int,
    output_stream~ : Int
  )
} derive(Debug, Eq)

///|
pub struct GlmTinyHybridBlockPlan {
  priv token_count : Int
  priv head_count : Int
  priv head_dimension : Int
  priv low_rank : Int
  priv sparse_top_k : Int
  priv intermediate_size : Int
  priv rms_epsilon : Double
  priv feed_forward : GlmTinyFeedForwardPlan
  priv composition : GlmTinyCompositionPlan
}

///|
pub fn GlmTinyHybridBlockPlan::new(
  token_count~ : Int,
  head_count~ : Int,
  head_dimension~ : Int,
  low_rank~ : Int,
  sparse_top_k~ : Int,
  intermediate_size~ : Int,
  rms_epsilon~ : Double,
  feed_forward~ : GlmTinyFeedForwardPlan,
  composition~ : GlmTinyCompositionPlan,
  limits~ : GlmTinyOracleLimits,
) -> GlmTinyHybridBlockPlan raise GlmHybridReferenceError {
  if head_count <= 0 ||
    head_dimension <= 0 ||
    head_count > limits.maximum_width / head_dimension {
    raise BoundsExceeded
  }
  let width = head_count * head_dimension
  if token_count <= 0 ||
    token_count > limits.maximum_tokens ||
    width > limits.maximum_width ||
    low_rank <= 0 ||
    low_rank > limits.maximum_low_rank ||
    sparse_top_k <= 0 ||
    sparse_top_k > token_count ||
    sparse_top_k > limits.maximum_selected ||
    intermediate_size <= 0 ||
    intermediate_size > limits.maximum_intermediate ||
    !finite(rms_epsilon) ||
    rms_epsilon <= 0.0 {
    raise BoundsExceeded
  }
  if token_count > limits.maximum_values / width {
    raise BoundsExceeded
  }
  match feed_forward {
    Dense => ()
    Moe(expert_count~, experts_per_token~, route_scale~, ..) =>
      if expert_count <= 0 ||
        expert_count > limits.maximum_experts ||
        experts_per_token <= 0 ||
        experts_per_token > expert_count ||
        experts_per_token > limits.maximum_selected ||
        !oracle_float_finite(route_scale) ||
        route_scale <= 0.0F {
        raise BoundsExceeded
      }
  }
  match composition {
    Residual => ()
    MultiStreamHyperConnection(stream_count~, branch_stream~, output_stream~) =>
      if stream_count <= 1 ||
        stream_count > limits.maximum_width ||
        stream_count > limits.maximum_values / (token_count * width) ||
        branch_stream < 0 ||
        branch_stream >= stream_count ||
        output_stream < 0 ||
        output_stream >= stream_count {
        raise BoundsExceeded
      }
  }
  {
    token_count,
    head_count,
    head_dimension,
    low_rank,
    sparse_top_k,
    intermediate_size,
    rms_epsilon,
    feed_forward,
    composition,
  }
}

///|
priv enum GlmTinyFeedForwardWeightStorage {
  DenseWeights(
    ReadOnlyArray[Double],
    ReadOnlyArray[Double],
    ReadOnlyArray[Double]
  )
  MoeWeights(
    ReadOnlyArray[Float],
    ReadOnlyArray[Float],
    ReadOnlyArray[Double],
    ReadOnlyArray[Double],
    ReadOnlyArray[Double]
  )
}

///|
pub struct GlmTinyFeedForwardWeights {
  priv storage : GlmTinyFeedForwardWeightStorage
}

///|
pub fn GlmTinyFeedForwardWeights::dense(
  gate~ : ArrayView[Double],
  up~ : ArrayView[Double],
  down~ : ArrayView[Double],
) -> GlmTinyFeedForwardWeights {
  {
    storage: DenseWeights(
      ReadOnlyArray::from_array(gate),
      ReadOnlyArray::from_array(up),
      ReadOnlyArray::from_array(down),
    ),
  }
}

///|
pub fn GlmTinyFeedForwardWeights::moe(
  router~ : ArrayView[Float],
  correction~ : ArrayView[Float],
  gate~ : ArrayView[Double],
  up~ : ArrayView[Double],
  down~ : ArrayView[Double],
) -> GlmTinyFeedForwardWeights {
  {
    storage: MoeWeights(
      ReadOnlyArray::from_array(router),
      ReadOnlyArray::from_array(correction),
      ReadOnlyArray::from_array(gate),
      ReadOnlyArray::from_array(up),
      ReadOnlyArray::from_array(down),
    ),
  }
}

///|
pub struct GlmTinyHybridBlockWeights {
  priv attention_norm : ReadOnlyArray[Double]
  priv query_a : ReadOnlyArray[Double]
  priv query_b : ReadOnlyArray[Double]
  priv key_a : ReadOnlyArray[Double]
  priv key_b : ReadOnlyArray[Double]
  priv value_a : ReadOnlyArray[Double]
  priv value_b : ReadOnlyArray[Double]
  priv attention_output : ReadOnlyArray[Double]
  priv ffn_norm : ReadOnlyArray[Double]
  priv feed_forward : GlmTinyFeedForwardWeights
  priv hyper_mixing : ReadOnlyArray[Double]
}

///|
pub fn GlmTinyHybridBlockWeights::new(
  attention_norm~ : ArrayView[Double],
  query_a~ : ArrayView[Double],
  query_b~ : ArrayView[Double],
  key_a~ : ArrayView[Double],
  key_b~ : ArrayView[Double],
  value_a~ : ArrayView[Double],
  value_b~ : ArrayView[Double],
  attention_output~ : ArrayView[Double],
  ffn_norm~ : ArrayView[Double],
  feed_forward~ : GlmTinyFeedForwardWeights,
  hyper_mixing~ : ArrayView[Double],
) -> GlmTinyHybridBlockWeights {
  {
    attention_norm: ReadOnlyArray::from_array(attention_norm),
    query_a: ReadOnlyArray::from_array(query_a),
    query_b: ReadOnlyArray::from_array(query_b),
    key_a: ReadOnlyArray::from_array(key_a),
    key_b: ReadOnlyArray::from_array(key_b),
    value_a: ReadOnlyArray::from_array(value_a),
    value_b: ReadOnlyArray::from_array(value_b),
    attention_output: ReadOnlyArray::from_array(attention_output),
    ffn_norm: ReadOnlyArray::from_array(ffn_norm),
    feed_forward,
    hyper_mixing: ReadOnlyArray::from_array(hyper_mixing),
  }
}

///|
pub struct GlmTinyHybridBlockResult {
  priv attention_branch : ReadOnlyArray[Double]
  priv feed_forward_branch : ReadOnlyArray[Double]
  priv output : ReadOnlyArray[Double]
  priv selected_experts : ReadOnlyArray[Int]
}
