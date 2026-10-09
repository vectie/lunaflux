///|
pub fn AdvancedDecoderShape::vocabulary_size(self : Self) -> Int {
  self.vocabulary_size
}

///|
pub fn AdvancedDecoderShape::hidden_size(self : Self) -> Int {
  self.hidden_size
}

///|
pub fn AdvancedDecoderShape::decoder_layer_count(self : Self) -> Int {
  self.decoder_layer_count
}

///|
pub fn AdvancedDecoderShape::attention_head_count(self : Self) -> Int {
  self.attention_head_count
}

///|
pub fn AdvancedDecoderShape::kv_head_count(self : Self) -> Int {
  self.kv_head_count
}

///|
pub fn AdvancedDecoderShape::attention_head_dimension(self : Self) -> Int {
  self.attention_head_dimension
}

///|
pub fn AdvancedDecoderShape::rope_head_dimension(self : Self) -> Int {
  self.rope_head_dimension
}

///|
pub fn AdvancedDecoderShape::query_low_rank(self : Self) -> Int {
  self.query_low_rank
}

///|
pub fn AdvancedDecoderShape::output_low_rank(self : Self) -> Int {
  self.output_low_rank
}

///|
pub fn AdvancedDecoderShape::output_group_count(self : Self) -> Int {
  self.output_group_count
}

///|
pub fn AdvancedDecoderShape::maximum_position_count(self : Self) -> Int {
  self.maximum_position_count
}

///|
pub fn LowRankAttentionPlan::query_rank(self : Self) -> Int {
  self.query_rank
}

///|
pub fn LowRankAttentionPlan::output_rank(self : Self) -> Int {
  self.output_rank
}

///|
pub fn LowRankAttentionPlan::output_group_count(self : Self) -> Int {
  self.output_group_count
}

///|
pub fn LowRankAttentionPlan::query_key_norm_epsilon(self : Self) -> Double {
  self.query_key_norm_epsilon
}

///|
pub fn PositionEncodingPlan::rope_theta(self : Self) -> Double {
  self.rope_theta
}

///|
pub fn PositionEncodingPlan::compressed_rope_theta(self : Self) -> Double {
  self.compressed_rope_theta
}

///|
pub fn PositionEncodingPlan::original_position_count(self : Self) -> Int {
  self.original_position_count
}

///|
pub fn PositionEncodingPlan::scaling_factor(self : Self) -> Double {
  self.scaling_factor
}

///|
pub fn PositionEncodingPlan::beta_fast(self : Self) -> Double {
  self.beta_fast
}

///|
pub fn PositionEncodingPlan::beta_slow(self : Self) -> Double {
  self.beta_slow
}

///|
pub fn HyperConnectionPlan::width(self : Self) -> Int {
  self.width
}

///|
pub fn HyperConnectionPlan::control_epsilon(self : Self) -> Double {
  self.control_epsilon
}

///|
/// Compatibility accessor for the control/Sinkhorn epsilon.
pub fn HyperConnectionPlan::epsilon(self : Self) -> Double {
  self.control_epsilon
}

///|
pub fn HyperConnectionPlan::normalization_epsilon(self : Self) -> Double {
  self.normalization_epsilon
}

///|
pub fn HyperConnectionPlan::sinkhorn_iterations(self : Self) -> Int {
  self.sinkhorn_iterations
}

///|
pub fn MoeExecutionPlan::routed_expert_count(self : Self) -> Int {
  self.routed_expert_count
}

///|
pub fn MoeExecutionPlan::shared_expert_count(self : Self) -> Int {
  self.shared_expert_count
}

///|
pub fn MoeExecutionPlan::experts_per_token(self : Self) -> Int {
  self.experts_per_token
}

///|
pub fn MoeExecutionPlan::intermediate_size(self : Self) -> Int {
  self.intermediate_size
}

///|
pub fn MoeExecutionPlan::routed_scaling_factor(self : Self) -> Double {
  self.routed_scaling_factor
}

///|
pub fn MoeExecutionPlan::normalize_selected_probabilities(self : Self) -> Bool {
  self.normalize_selected_probabilities
}

///|
pub fn MoeExecutionPlan::score_function(self : Self) -> ExpertScoreFunction {
  self.score_function
}

///|
pub fn MoeExecutionPlan::selection_method(self : Self) -> ExpertSelectionMethod {
  self.selection_method
}

///|
pub fn MoeExecutionPlan::activation(self : Self) -> ExpertActivation {
  self.activation
}

///|
pub fn AdvancedDecoderLayerPlan::layer_index(self : Self) -> Int {
  self.layer_index
}

///|
pub fn AdvancedDecoderLayerPlan::attention(
  self : Self,
) -> AdvancedAttentionPlan {
  self.attention
}

///|
pub fn DsparkExecutionPlan::block_size(self : Self) -> Int {
  self.block_size
}

///|
pub fn DsparkExecutionPlan::noise_token_id(self : Self) -> Int {
  self.noise_token_id
}

///|
pub fn DsparkExecutionPlan::target_layer_ids(self : Self) -> ReadOnlyArray[Int] {
  self.target_layer_ids
}

///|
pub fn DsparkExecutionPlan::markov_rank(self : Self) -> Int {
  self.markov_rank
}

///|
pub fn AdvancedDecoderExecutionPlan::schema_version(self : Self) -> Int {
  self.schema_version
}

///|
pub fn AdvancedDecoderExecutionPlan::profile_identity(self : Self) -> String {
  self.profile_identity
}

///|
pub fn AdvancedDecoderExecutionPlan::content_digest(
  self : Self,
) -> @spec.ContentDigest {
  self.content_digest
}

///|
pub fn AdvancedDecoderExecutionPlan::shape(self : Self) -> AdvancedDecoderShape {
  self.shape
}

///|
pub fn AdvancedDecoderExecutionPlan::low_rank_attention(
  self : Self,
) -> LowRankAttentionPlan {
  self.low_rank_attention
}

///|
pub fn AdvancedDecoderExecutionPlan::position_encoding(
  self : Self,
) -> PositionEncodingPlan {
  self.position_encoding
}

///|
pub fn AdvancedDecoderExecutionPlan::hyper_connection(
  self : Self,
) -> HyperConnectionPlan {
  self.hyper_connection
}

///|
pub fn AdvancedDecoderExecutionPlan::moe(self : Self) -> MoeExecutionPlan {
  self.moe
}

///|
pub fn AdvancedDecoderExecutionPlan::layers(
  self : Self,
) -> ReadOnlyArray[AdvancedDecoderLayerPlan] {
  self.layers
}

///|
pub fn AdvancedDecoderExecutionPlan::auxiliary_prediction(
  self : Self,
) -> AuxiliaryPredictionPlan {
  self.auxiliary_prediction
}

///|
pub fn AdvancedDecoderExecutionPlan::plan_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.plan_digest
}
