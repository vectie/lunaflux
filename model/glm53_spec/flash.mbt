///|
/// Exact semantic profile of the official multimodal GLM-5.3-Flash model.
///
/// This is not a smaller instance of the full model. It has a vision tower,
/// manifold-constrained hyper-connections, and a fixed hybrid KDA/DSA layer
/// schedule. Keeping a distinct owner prevents accidental dense-plan reuse.
pub struct Glm53FlashSpec {
  priv profile : Glm53ProfileId
} derive(Debug, Eq)

///|
pub fn Glm53FlashSpec::official() -> Glm53FlashSpec {
  { profile: FlashMultimodal, }
}

///|
pub fn Glm53FlashSpec::profile(self : Self) -> Glm53ProfileId {
  self.profile
}

///|
pub fn Glm53FlashSpec::architecture(self : Self) -> String {
  ignore(self)
  "Glm5NextForConditionalGeneration"
}

///|
pub fn Glm53FlashSpec::model_type(self : Self) -> String {
  ignore(self)
  "glm5_next"
}

///|
pub fn Glm53FlashSpec::text_model_type(self : Self) -> String {
  ignore(self)
  "glm5_next_text"
}

///|
pub fn Glm53FlashSpec::vision_model_type(self : Self) -> String {
  ignore(self)
  "glm5_next_vision"
}

///|
pub fn Glm53FlashSpec::vocabulary_size(self : Self) -> Int {
  ignore(self)
  154880
}

///|
pub fn Glm53FlashSpec::hidden_size(self : Self) -> Int {
  ignore(self)
  4096
}

///|
pub fn Glm53FlashSpec::dense_intermediate_size(self : Self) -> Int {
  ignore(self)
  12288
}

///|
pub fn Glm53FlashSpec::moe_intermediate_size(self : Self) -> Int {
  ignore(self)
  2048
}

///|
pub fn Glm53FlashSpec::layer_count(self : Self) -> Int {
  ignore(self)
  45
}

///|
pub fn Glm53FlashSpec::attention_head_count(self : Self) -> Int {
  ignore(self)
  64
}

///|
pub fn Glm53FlashSpec::kv_head_count(self : Self) -> Int {
  ignore(self)
  64
}

///|
pub fn Glm53FlashSpec::q_lora_rank(self : Self) -> Int {
  ignore(self)
  1536
}

///|
pub fn Glm53FlashSpec::kv_lora_rank(self : Self) -> Int {
  ignore(self)
  512
}

///|
pub fn Glm53FlashSpec::qk_nope_head_dimension(self : Self) -> Int {
  ignore(self)
  256
}

///|
pub fn Glm53FlashSpec::qk_rope_head_dimension(self : Self) -> Int {
  ignore(self)
  0
}

///|
pub fn Glm53FlashSpec::value_head_dimension(self : Self) -> Int {
  ignore(self)
  256
}

///|
pub fn Glm53FlashSpec::index_head_count(self : Self) -> Int {
  ignore(self)
  32
}

///|
pub fn Glm53FlashSpec::index_head_dimension(self : Self) -> Int {
  ignore(self)
  128
}

///|
pub fn Glm53FlashSpec::index_topk(self : Self) -> Int {
  ignore(self)
  2048
}

///|
pub fn Glm53FlashSpec::index_kpool(self : Self) -> Int {
  ignore(self)
  4
}

///|
pub fn Glm53FlashSpec::linear_head_count(self : Self) -> Int {
  ignore(self)
  64
}

///|
pub fn Glm53FlashSpec::linear_head_dimension(self : Self) -> Int {
  ignore(self)
  128
}

///|
pub fn Glm53FlashSpec::linear_conv_kernel_size(self : Self) -> Int {
  ignore(self)
  4
}

///|
pub fn Glm53FlashSpec::hyper_connection_streams(self : Self) -> Int {
  ignore(self)
  4
}

///|
pub fn Glm53FlashSpec::hyper_connection_sinkhorn_iterations(self : Self) -> Int {
  ignore(self)
  20
}

///|
pub fn Glm53FlashSpec::hyper_connection_epsilon(self : Self) -> Double {
  ignore(self)
  0.000001
}

///|
pub fn Glm53FlashSpec::context_length(self : Self) -> Int {
  ignore(self)
  1048576
}

///|
pub fn Glm53FlashSpec::rms_norm_epsilon(self : Self) -> Double {
  ignore(self)
  0.00001
}

///|
pub fn Glm53FlashSpec::routed_scaling_factor(self : Self) -> Double {
  ignore(self)
  2.5
}

///|
pub fn Glm53FlashSpec::expert_group_count(self : Self) -> Int {
  ignore(self)
  1
}

///|
pub fn Glm53FlashSpec::selected_expert_group_count(self : Self) -> Int {
  ignore(self)
  1
}

///|
pub fn Glm53FlashSpec::expert_group_score_topk(self : Self) -> Int {
  ignore(self)
  2
}

///|
pub fn Glm53FlashSpec::normalizes_selected_expert_probabilities(
  self : Self,
) -> Bool {
  ignore(self)
  true
}

///|
pub fn Glm53FlashSpec::uses_float32_moe_router(self : Self) -> Bool {
  ignore(self)
  true
}

///|
pub fn Glm53FlashSpec::uses_moe_correction_bias(self : Self) -> Bool {
  ignore(self)
  true
}

///|
pub fn Glm53FlashSpec::swiglu_limit(self : Self) -> Double {
  ignore(self)
  10.0
}

///|
pub fn Glm53FlashSpec::linear_gate_lower_bound(self : Self) -> Double {
  ignore(self)
  -5.0
}

///|
pub fn Glm53FlashSpec::index_pool_compress(self : Self) -> Bool {
  ignore(self)
  true
}

///|
pub fn Glm53FlashSpec::index_pool_always_select_tail(self : Self) -> Bool {
  ignore(self)
  true
}

///|
pub fn Glm53FlashSpec::indexer_rope_interleave(self : Self) -> Bool {
  ignore(self)
  true
}

///|
pub fn Glm53FlashSpec::index_share_for_mtp_iteration(self : Self) -> Bool {
  ignore(self)
  true
}

///|
pub fn Glm53FlashSpec::routed_expert_count(self : Self) -> Int {
  ignore(self)
  288
}

///|
pub fn Glm53FlashSpec::shared_expert_count(self : Self) -> Int {
  ignore(self)
  1
}

///|
pub fn Glm53FlashSpec::experts_per_token(self : Self) -> Int {
  ignore(self)
  8
}

///|
pub fn Glm53FlashSpec::mtp_layer_count(self : Self) -> Int {
  ignore(self)
  1
}

///|
pub fn Glm53FlashSpec::vision_layer_count(self : Self) -> Int {
  ignore(self)
  24
}

///|
pub fn Glm53FlashSpec::vision_hidden_size(self : Self) -> Int {
  ignore(self)
  1024
}

///|
pub fn Glm53FlashSpec::vision_intermediate_size(self : Self) -> Int {
  ignore(self)
  4096
}

///|
pub fn Glm53FlashSpec::vision_head_count(self : Self) -> Int {
  ignore(self)
  16
}

///|
pub fn Glm53FlashSpec::vision_patch_size(self : Self) -> Int {
  ignore(self)
  14
}

///|
pub fn Glm53FlashSpec::vision_temporal_patch_size(self : Self) -> Int {
  ignore(self)
  2
}

///|
pub fn Glm53FlashSpec::vision_spatial_merge_size(self : Self) -> Int {
  ignore(self)
  2
}

///|
pub fn Glm53FlashSpec::vision_projection_size(self : Self) -> Int {
  ignore(self)
  4096
}

///|
pub fn Glm53FlashSpec::vision_projection_intermediate_size(self : Self) -> Int {
  ignore(self)
  10240
}

///|
pub fn Glm53FlashSpec::image_token_id(self : Self) -> Int {
  ignore(self)
  154854
}

///|
pub fn Glm53FlashSpec::video_token_id(self : Self) -> Int {
  ignore(self)
  154855
}

///|
pub fn Glm53FlashSpec::image_start_token_id(self : Self) -> Int {
  ignore(self)
  154830
}

///|
pub fn Glm53FlashSpec::image_end_token_id(self : Self) -> Int {
  ignore(self)
  154831
}

///|
pub fn Glm53FlashSpec::video_start_token_id(self : Self) -> Int {
  ignore(self)
  154832
}

///|
pub fn Glm53FlashSpec::video_end_token_id(self : Self) -> Int {
  ignore(self)
  154833
}

///|
pub fn Glm53FlashSpec::attention_kind(
  self : Self,
  layer : Int,
) -> Glm53AttentionKind raise Glm53SpecError {
  if layer < 0 || layer >= self.layer_count() {
    raise InvalidLayerIndex
  }
  if layer % 4 == 3 {
    DeepSeekSparseAttention
  } else {
    KimiDeltaAttention
  }
}

///|
pub fn Glm53FlashSpec::feed_forward_kind(
  self : Self,
  layer : Int,
) -> Glm53FeedForwardKind raise Glm53SpecError {
  if layer < 0 || layer >= self.layer_count() {
    raise InvalidLayerIndex
  }
  if layer < 3 {
    DenseSwiGlu
  } else {
    RoutedMixtureOfExperts
  }
}
