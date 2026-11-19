///|
pub fn GlmSparseIndexPoolingPlan::kernel_size(self : Self) -> Int32 {
  self.kernel_size
}

///|
pub fn GlmSparseIndexPoolingPlan::compress(self : Self) -> Bool {
  self.compress
}

///|
pub fn GlmSparseIndexPoolingPlan::always_select_tail(self : Self) -> Bool {
  self.always_select_tail
}

///|
pub fn GlmDynamicSparseAttentionPlan::query_head_count(self : Self) -> Int32 {
  self.query_head_count
}

///|
pub fn GlmDynamicSparseAttentionPlan::kv_head_count(self : Self) -> Int32 {
  self.kv_head_count
}

///|
pub fn GlmDynamicSparseAttentionPlan::query_lora_rank(self : Self) -> Int32 {
  self.query_lora_rank
}

///|
pub fn GlmDynamicSparseAttentionPlan::kv_lora_rank(self : Self) -> Int32 {
  self.kv_lora_rank
}

///|
pub fn GlmDynamicSparseAttentionPlan::qk_nope_head_dimension(
  self : Self,
) -> Int32 {
  self.qk_nope_head_dimension
}

///|
pub fn GlmDynamicSparseAttentionPlan::qk_rope_head_dimension(
  self : Self,
) -> Int32 {
  self.qk_rope_head_dimension
}

///|
pub fn GlmDynamicSparseAttentionPlan::value_head_dimension(
  self : Self,
) -> Int32 {
  self.value_head_dimension
}

///|
pub fn GlmDynamicSparseAttentionPlan::rms_norm_epsilon(self : Self) -> Double {
  self.rms_norm_epsilon
}

///|
pub fn GlmDynamicSparseAttentionPlan::rope_theta(self : Self) -> Double? {
  self.rope_theta
}

///|
pub fn GlmDynamicSparseAttentionPlan::rope_interleave(self : Self) -> Bool {
  self.rope_interleave
}

///|
pub fn GlmDynamicSparseAttentionPlan::index_head_count(self : Self) -> Int32 {
  self.index_head_count
}

///|
pub fn GlmDynamicSparseAttentionPlan::index_head_dimension(
  self : Self,
) -> Int32 {
  self.index_head_dimension
}

///|
pub fn GlmDynamicSparseAttentionPlan::index_topk(self : Self) -> Int32 {
  self.index_topk
}

///|
pub fn GlmDynamicSparseAttentionPlan::indexer_rope_interleave(
  self : Self,
) -> Bool {
  self.indexer_rope_interleave
}

///|
pub fn GlmDynamicSparseAttentionPlan::index_share_for_mtp_iteration(
  self : Self,
) -> Bool {
  self.index_share_for_mtp_iteration
}

///|
pub fn GlmDynamicSparseAttentionPlan::index_pooling(
  self : Self,
) -> GlmSparseIndexPoolingPlan? {
  self.index_pooling
}

///|
pub fn GlmKdaExecutionPlan::head_count(self : Self) -> Int32 {
  self.head_count
}

///|
pub fn GlmKdaExecutionPlan::head_dimension(self : Self) -> Int32 {
  self.head_dimension
}

///|
pub fn GlmKdaExecutionPlan::short_convolution_kernel_size(self : Self) -> Int32 {
  self.short_convolution_kernel_size
}

///|
pub fn GlmKdaExecutionPlan::gate_lower_bound(self : Self) -> Double {
  self.gate_lower_bound
}

///|
pub fn GlmKdaExecutionPlan::projected_width(self : Self) -> Int32 {
  self.head_count * self.head_dimension
}

///|
pub fn GlmKdaExecutionPlan::recurrent_matrix_elements_per_sequence(
  self : Self,
) -> Int64 {
  Int64::from_int(self.head_count) *
  Int64::from_int(self.head_dimension) *
  Int64::from_int(self.head_dimension)
}

///|
pub fn GlmKdaExecutionPlan::convolution_history_tokens(self : Self) -> Int32 {
  self.short_convolution_kernel_size - 1
}

///|
pub fn GlmHybridLayerPlan::layer_index(self : Self) -> Int32 {
  self.layer_index
}

///|
pub fn GlmHybridLayerPlan::attention(self : Self) -> GlmLayerAttentionPlan {
  self.attention
}

///|
pub fn GlmHybridLayerPlan::feed_forward(self : Self) -> GlmLayerFeedForwardPlan {
  self.feed_forward
}

///|
pub fn GlmVisionConditioningPlan::layer_count(self : Self) -> Int32 {
  self.layer_count
}

///|
pub fn GlmVisionConditioningPlan::hidden_size(self : Self) -> Int32 {
  self.hidden_size
}

///|
pub fn GlmVisionConditioningPlan::intermediate_size(self : Self) -> Int32 {
  self.intermediate_size
}

///|
pub fn GlmVisionConditioningPlan::attention_head_count(self : Self) -> Int32 {
  self.attention_head_count
}

///|
pub fn GlmVisionConditioningPlan::spatial_patch_size(self : Self) -> Int32 {
  self.spatial_patch_size
}

///|
pub fn GlmVisionConditioningPlan::temporal_patch_size(self : Self) -> Int32 {
  self.temporal_patch_size
}

///|
pub fn GlmVisionConditioningPlan::spatial_merge_size(self : Self) -> Int32 {
  self.spatial_merge_size
}

///|
pub fn GlmVisionConditioningPlan::decoder_projection_size(self : Self) -> Int32 {
  self.decoder_projection_size
}

///|
pub fn GlmVisionConditioningPlan::projection_intermediate_size(
  self : Self,
) -> Int32 {
  self.projection_intermediate_size
}

///|
pub fn GlmVisionConditioningPlan::image_token_id(self : Self) -> Int32 {
  self.image_token_id
}

///|
pub fn GlmVisionConditioningPlan::video_token_id(self : Self) -> Int32 {
  self.video_token_id
}

///|
pub fn GlmVisionConditioningPlan::image_start_token_id(self : Self) -> Int32 {
  self.image_start_token_id
}

///|
pub fn GlmVisionConditioningPlan::image_end_token_id(self : Self) -> Int32 {
  self.image_end_token_id
}

///|
pub fn GlmVisionConditioningPlan::video_start_token_id(self : Self) -> Int32 {
  self.video_start_token_id
}

///|
pub fn GlmVisionConditioningPlan::video_end_token_id(self : Self) -> Int32 {
  self.video_end_token_id
}
