///|
pub fn GlmSparseIndexPoolingPlan::new(
  kernel_size~ : Int,
  compress~ : Bool,
  always_select_tail~ : Bool,
) -> GlmSparseIndexPoolingPlan raise GlmHybridExecutionPlanError {
  if kernel_size <= 0 {
    raise InvalidSparseAttentionGeometry
  }
  { kernel_size, compress, always_select_tail, }
}

///|
pub fn GlmDynamicSparseAttentionPlan::new(
  query_head_count~ : Int,
  kv_head_count~ : Int,
  query_lora_rank~ : Int,
  kv_lora_rank~ : Int,
  qk_nope_head_dimension~ : Int,
  qk_rope_head_dimension~ : Int,
  value_head_dimension~ : Int,
  rms_norm_epsilon~ : Double,
  rope_theta~ : Double?,
  rope_interleave~ : Bool,
  index_head_count~ : Int,
  index_head_dimension~ : Int,
  index_topk~ : Int,
  indexer_rope_interleave~ : Bool,
  index_share_for_mtp_iteration~ : Bool,
  index_pooling~ : GlmSparseIndexPoolingPlan?,
) -> GlmDynamicSparseAttentionPlan raise GlmHybridExecutionPlanError {
  let rope_valid = match (qk_rope_head_dimension, rope_theta) {
    (0, None) => !rope_interleave
    (dimension, Some(theta)) =>
      dimension > 0 &&
      !theta.is_nan() &&
      !theta.is_inf() &&
      theta > 0.0 &&
      theta <= 1.0e12
    _ => false
  }
  if query_head_count <= 0 ||
    kv_head_count <= 0 ||
    kv_head_count > query_head_count ||
    query_head_count % kv_head_count != 0 ||
    query_lora_rank <= 0 ||
    kv_lora_rank <= 0 ||
    qk_nope_head_dimension <= 0 ||
    qk_rope_head_dimension < 0 ||
    value_head_dimension <= 0 ||
    rms_norm_epsilon.is_nan() ||
    rms_norm_epsilon.is_inf() ||
    rms_norm_epsilon <= 0.0 ||
    rms_norm_epsilon > 1.0 ||
    index_head_count <= 0 ||
    index_head_count > query_head_count ||
    query_head_count % index_head_count != 0 ||
    index_head_dimension <= 0 ||
    index_topk <= 0 ||
    !rope_valid {
    raise InvalidSparseAttentionGeometry
  }
  match index_pooling {
    Some(pooling) =>
      if pooling.kernel_size() > index_head_dimension {
        raise InvalidSparseAttentionGeometry
      }
    None => ()
  }
  {
    query_head_count,
    kv_head_count,
    query_lora_rank,
    kv_lora_rank,
    qk_nope_head_dimension,
    qk_rope_head_dimension,
    value_head_dimension,
    rms_norm_epsilon,
    rope_theta,
    rope_interleave,
    index_head_count,
    index_head_dimension,
    index_topk,
    indexer_rope_interleave,
    index_share_for_mtp_iteration,
    index_pooling,
  }
}

///|
pub fn GlmKdaExecutionPlan::new(
  head_count~ : Int,
  head_dimension~ : Int,
  short_convolution_kernel_size~ : Int,
  gate_lower_bound~ : Double,
) -> GlmKdaExecutionPlan raise GlmHybridExecutionPlanError {
  if head_count <= 0 ||
    head_dimension <= 0 ||
    short_convolution_kernel_size <= 0 ||
    gate_lower_bound.is_nan() ||
    gate_lower_bound.is_inf() ||
    gate_lower_bound > 0.0 {
    raise InvalidKdaGeometry
  }
  {
    head_count,
    head_dimension,
    short_convolution_kernel_size,
    gate_lower_bound,
  }
}

///|
pub fn GlmHybridLayerPlan::new(
  layer_index~ : Int,
  attention~ : GlmLayerAttentionPlan,
  feed_forward~ : GlmLayerFeedForwardPlan,
) -> GlmHybridLayerPlan raise GlmHybridExecutionPlanError {
  if layer_index < 0 {
    raise InvalidLayerOrder
  }
  { layer_index, attention, feed_forward, }
}

///|
pub fn GlmVisionConditioningPlan::new(
  layer_count~ : Int,
  hidden_size~ : Int,
  intermediate_size~ : Int,
  attention_head_count~ : Int,
  spatial_patch_size~ : Int,
  temporal_patch_size~ : Int,
  spatial_merge_size~ : Int,
  decoder_projection_size~ : Int,
  projection_intermediate_size~ : Int,
  vocabulary_size~ : Int,
  image_token_id~ : Int,
  video_token_id~ : Int,
  image_start_token_id~ : Int,
  image_end_token_id~ : Int,
  video_start_token_id~ : Int,
  video_end_token_id~ : Int,
) -> GlmVisionConditioningPlan raise GlmHybridExecutionPlanError {
  let tokens = [
    image_token_id, video_token_id, image_start_token_id, image_end_token_id, video_start_token_id,
    video_end_token_id,
  ]
  let mut distinct = true
  for left in 0..<tokens.length() {
    if tokens[left] < 0 || tokens[left] >= vocabulary_size {
      distinct = false
    }
    for right in (left + 1)..<tokens.length() {
      if tokens[left] == tokens[right] {
        distinct = false
      }
    }
  }
  if layer_count <= 0 ||
    hidden_size <= 0 ||
    intermediate_size <= 0 ||
    attention_head_count <= 0 ||
    hidden_size % attention_head_count != 0 ||
    spatial_patch_size <= 0 ||
    temporal_patch_size <= 0 ||
    spatial_merge_size <= 0 ||
    decoder_projection_size <= 0 ||
    projection_intermediate_size <= 0 ||
    vocabulary_size <= 0 ||
    !distinct {
    raise InvalidVisionConditioning
  }
  {
    layer_count,
    hidden_size,
    intermediate_size,
    attention_head_count,
    spatial_patch_size,
    temporal_patch_size,
    spatial_merge_size,
    decoder_projection_size,
    projection_intermediate_size,
    image_token_id,
    video_token_id,
    image_start_token_id,
    image_end_token_id,
    video_start_token_id,
    video_end_token_id,
  }
}
