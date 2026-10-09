///|
pub fn AdvancedDecoderShape::new(
  vocabulary_size~ : Int,
  hidden_size~ : Int,
  decoder_layer_count~ : Int,
  attention_head_count~ : Int,
  kv_head_count~ : Int,
  attention_head_dimension~ : Int,
  rope_head_dimension~ : Int,
  query_low_rank~ : Int,
  output_low_rank~ : Int,
  output_group_count~ : Int,
  maximum_position_count~ : Int,
) -> AdvancedDecoderShape raise AdvancedDecoderPlanError {
  for
    entry in [
      (vocabulary_size, VocabularySize),
      (hidden_size, HiddenSize),
      (decoder_layer_count, DecoderLayerCount),
      (attention_head_count, AttentionHeadCount),
      (kv_head_count, KeyValueHeadCount),
      (attention_head_dimension, AttentionHeadDimension),
      (rope_head_dimension, RopeHeadDimension),
      (query_low_rank, QueryLowRank),
      (output_low_rank, OutputLowRank),
      (output_group_count, OutputGroupCount),
      (maximum_position_count, MaximumPositionCount),
    ] {
    let (value, dimension) = entry
    if value <= 0 {
      raise InvalidDimension(dimension)
    }
  }
  if kv_head_count > attention_head_count ||
    attention_head_count % kv_head_count != 0 ||
    rope_head_dimension > attention_head_dimension ||
    output_group_count > attention_head_count ||
    attention_head_count % output_group_count != 0 {
    raise InvalidAttentionGeometry
  }
  {
    vocabulary_size,
    hidden_size,
    decoder_layer_count,
    attention_head_count,
    kv_head_count,
    attention_head_dimension,
    rope_head_dimension,
    query_low_rank,
    output_low_rank,
    output_group_count,
    maximum_position_count,
  }
}

///|
pub fn LowRankAttentionPlan::new(
  query_rank~ : Int,
  output_rank~ : Int,
  output_group_count~ : Int,
  query_key_norm_epsilon~ : Double,
) -> LowRankAttentionPlan raise AdvancedDecoderPlanError {
  if query_rank <= 0 ||
    output_rank <= 0 ||
    output_group_count <= 0 ||
    query_key_norm_epsilon.is_nan() ||
    query_key_norm_epsilon.is_inf() ||
    query_key_norm_epsilon <= 0.0 ||
    query_key_norm_epsilon > 1.0 {
    raise InvalidAttentionGeometry
  }
  { query_rank, output_rank, output_group_count, query_key_norm_epsilon, }
}

///|
pub fn PositionEncodingPlan::new(
  rope_theta~ : Double,
  compressed_rope_theta~ : Double,
  original_position_count~ : Int,
  scaling_factor~ : Double,
  beta_fast~ : Double,
  beta_slow~ : Double,
) -> PositionEncodingPlan raise AdvancedDecoderPlanError {
  if !finite_positive(rope_theta) ||
    !finite_positive(compressed_rope_theta) ||
    original_position_count <= 0 ||
    !finite_positive(scaling_factor) ||
    !finite_positive(beta_fast) ||
    !finite_positive(beta_slow) ||
    compressed_rope_theta < rope_theta ||
    beta_fast < beta_slow {
    raise InvalidPositionEncoding
  }
  {
    rope_theta,
    compressed_rope_theta,
    original_position_count,
    scaling_factor,
    beta_fast,
    beta_slow,
  }
}

///|
fn finite_positive(value : Double) -> Bool {
  !value.is_nan() && !value.is_inf() && value > 0.0
}

///|
pub fn HyperConnectionPlan::new(
  width~ : Int,
  control_epsilon~ : Double,
  normalization_epsilon~ : Double,
  sinkhorn_iterations~ : Int,
) -> HyperConnectionPlan raise AdvancedDecoderPlanError {
  if width <= 1 ||
    !finite_positive(control_epsilon) ||
    control_epsilon > 1.0 ||
    !finite_positive(normalization_epsilon) ||
    normalization_epsilon > 1.0 ||
    sinkhorn_iterations <= 0 {
    raise InvalidHyperConnection
  }
  { width, control_epsilon, normalization_epsilon, sinkhorn_iterations, }
}
