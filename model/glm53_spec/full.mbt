///|
/// Exact semantic profile of the official full GLM-5.3 base model.
///
/// GLM-5.3 uses the same base model as GLM-5.2. This type deliberately
/// exposes only official, versioned constants: a neighboring GLM-MoE-DSA
/// checkpoint cannot enter by supplying merely plausible dimensions.
pub struct Glm53FullSpec {
  priv profile : Glm53ProfileId
} derive(Debug, Eq)

///|
pub fn Glm53FullSpec::official() -> Glm53FullSpec {
  { profile: FullText, }
}

///|
pub fn Glm53FullSpec::profile(self : Self) -> Glm53ProfileId {
  self.profile
}

///|
pub fn Glm53FullSpec::architecture(self : Self) -> String {
  ignore(self)
  "GlmMoeDsaForCausalLM"
}

///|
pub fn Glm53FullSpec::model_type(self : Self) -> String {
  ignore(self)
  "glm_moe_dsa"
}

///|
pub fn Glm53FullSpec::vocabulary_size(self : Self) -> Int {
  ignore(self)
  154880
}

///|
pub fn Glm53FullSpec::hidden_size(self : Self) -> Int {
  ignore(self)
  6144
}

///|
pub fn Glm53FullSpec::dense_intermediate_size(self : Self) -> Int {
  ignore(self)
  12288
}

///|
pub fn Glm53FullSpec::moe_intermediate_size(self : Self) -> Int {
  ignore(self)
  2048
}

///|
pub fn Glm53FullSpec::layer_count(self : Self) -> Int {
  ignore(self)
  78
}

///|
pub fn Glm53FullSpec::attention_head_count(self : Self) -> Int {
  ignore(self)
  64
}

///|
pub fn Glm53FullSpec::kv_head_count(self : Self) -> Int {
  ignore(self)
  64
}

///|
pub fn Glm53FullSpec::q_lora_rank(self : Self) -> Int {
  ignore(self)
  2048
}

///|
pub fn Glm53FullSpec::kv_lora_rank(self : Self) -> Int {
  ignore(self)
  512
}

///|
pub fn Glm53FullSpec::qk_nope_head_dimension(self : Self) -> Int {
  ignore(self)
  192
}

///|
pub fn Glm53FullSpec::qk_rope_head_dimension(self : Self) -> Int {
  ignore(self)
  64
}

///|
pub fn Glm53FullSpec::qk_head_dimension(self : Self) -> Int {
  self.qk_nope_head_dimension() + self.qk_rope_head_dimension()
}

///|
pub fn Glm53FullSpec::value_head_dimension(self : Self) -> Int {
  ignore(self)
  256
}

///|
pub fn Glm53FullSpec::index_head_count(self : Self) -> Int {
  ignore(self)
  32
}

///|
pub fn Glm53FullSpec::index_head_dimension(self : Self) -> Int {
  ignore(self)
  128
}

///|
pub fn Glm53FullSpec::index_topk(self : Self) -> Int {
  ignore(self)
  2048
}

///|
pub fn Glm53FullSpec::context_length(self : Self) -> Int {
  ignore(self)
  1048576
}

///|
pub fn Glm53FullSpec::routed_expert_count(self : Self) -> Int {
  ignore(self)
  256
}

///|
pub fn Glm53FullSpec::shared_expert_count(self : Self) -> Int {
  ignore(self)
  1
}

///|
pub fn Glm53FullSpec::experts_per_token(self : Self) -> Int {
  ignore(self)
  8
}

///|
pub fn Glm53FullSpec::expert_group_count(self : Self) -> Int {
  ignore(self)
  1
}

///|
pub fn Glm53FullSpec::selected_expert_group_count(self : Self) -> Int {
  ignore(self)
  1
}

///|
pub fn Glm53FullSpec::expert_group_score_topk(self : Self) -> Int {
  ignore(self)
  2
}

///|
pub fn Glm53FullSpec::mtp_layer_count(self : Self) -> Int {
  ignore(self)
  1
}

///|
pub fn Glm53FullSpec::rope_theta(self : Self) -> Double {
  ignore(self)
  8000000.0
}

///|
pub fn Glm53FullSpec::rms_norm_epsilon(self : Self) -> Double {
  ignore(self)
  0.00001
}

///|
pub fn Glm53FullSpec::routed_scaling_factor(self : Self) -> Double {
  ignore(self)
  2.5
}

///|
pub fn Glm53FullSpec::normalizes_selected_expert_probabilities(
  self : Self,
) -> Bool {
  ignore(self)
  true
}

///|
pub fn Glm53FullSpec::uses_float32_moe_router(self : Self) -> Bool {
  ignore(self)
  true
}

///|
pub fn Glm53FullSpec::uses_moe_correction_bias(self : Self) -> Bool {
  ignore(self)
  true
}

///|
pub fn Glm53FullSpec::rope_interleave(self : Self) -> Bool {
  ignore(self)
  true
}

///|
pub fn Glm53FullSpec::indexer_rope_interleave(self : Self) -> Bool {
  ignore(self)
  true
}

///|
pub fn Glm53FullSpec::index_share_for_mtp_iteration(self : Self) -> Bool {
  ignore(self)
  true
}

///|
pub fn Glm53FullSpec::attention_kind(
  self : Self,
  layer : Int,
) -> Glm53AttentionKind raise Glm53SpecError {
  if layer < 0 || layer >= self.layer_count() {
    raise InvalidLayerIndex
  }
  DeepSeekSparseAttention
}

///|
pub fn Glm53FullSpec::feed_forward_kind(
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

///|
/// `Full` layers compute DSA top-k; `Shared` layers reuse the preceding full
/// layer. The official schedule is FFFSSSFSSS... through layer 77.
pub fn Glm53FullSpec::computes_index_topk(
  self : Self,
  layer : Int,
) -> Bool raise Glm53SpecError {
  if layer < 0 || layer >= self.layer_count() {
    raise InvalidLayerIndex
  }
  layer < 3 || (layer - 2) % 4 == 0
}
