///|
/// Immutable architecture geometry shared by all decoder layers.
pub struct AdvancedDecoderShape {
  priv vocabulary_size : Int
  priv hidden_size : Int
  priv decoder_layer_count : Int
  priv attention_head_count : Int
  priv kv_head_count : Int
  priv attention_head_dimension : Int
  priv rope_head_dimension : Int
  priv query_low_rank : Int
  priv output_low_rank : Int
  priv output_group_count : Int
  priv maximum_position_count : Int
} derive(Debug, Eq)

///|
/// Low-rank attention projection and normalization semantics.
pub struct LowRankAttentionPlan {
  priv query_rank : Int
  priv output_rank : Int
  priv output_group_count : Int
  priv query_key_norm_epsilon : Double
} derive(Debug, Eq)

///|
/// Architecture-neutral position encoding with a distinct theta for indexed
/// compressed attention.
pub struct PositionEncodingPlan {
  priv rope_theta : Double
  priv compressed_rope_theta : Double
  priv original_position_count : Int
  priv scaling_factor : Double
  priv beta_fast : Double
  priv beta_slow : Double
} derive(Debug, Eq)

///|
/// Multi-stream residual mixing applied around every decoder block.
pub struct HyperConnectionPlan {
  priv width : Int
  priv control_epsilon : Double
  priv normalization_epsilon : Double
  priv sinkhorn_iterations : Int
} derive(Debug, Eq)

///|
/// Architecture-neutral expert gate scoring semantics.
pub(all) enum ExpertScoreFunction {
  SqrtSoftplus
  Sigmoid
} derive(Debug, Eq)

///|
/// Architecture-neutral routed-expert selection semantics.
pub(all) enum ExpertSelectionMethod {
  NoAuxiliaryLossTokenChoice
  TokenHashThenNoAuxiliaryLossTokenChoice(hash_layer_count~ : Int)
} derive(Debug, Eq)

///|
/// Expert activation semantics. The finite limit is part of the plan identity.
pub(all) enum ExpertActivation {
  SwiGlu
  ClampedSwiGlu(Double)
} derive(Debug, Eq)

///|
/// Immutable routed and shared-expert execution semantics.
pub struct MoeExecutionPlan {
  priv routed_expert_count : Int
  priv shared_expert_count : Int
  priv experts_per_token : Int
  priv intermediate_size : Int
  priv routed_scaling_factor : Double
  priv normalize_selected_probabilities : Bool
  priv score_function : ExpertScoreFunction
  priv selection_method : ExpertSelectionMethod
  priv activation : ExpertActivation
} derive(Debug, Eq)

///|
/// Per-layer attention kind. Both variants retain their complete geometry;
/// no family identifier is needed at execution-planning time.
pub(all) enum AdvancedAttentionPlan {
  SlidingWindowCausal(window_size~ : Int)
  IndexedCompressed(
    compression_ratio~ : Int,
    index_head_count~ : Int,
    index_head_dimension~ : Int,
    index_top_k~ : Int
  )
  DeterministicCompressed(compression_ratio~ : Int, window_size~ : Int)
} derive(Debug, Eq)

///|
/// One decoder layer in exact execution order.
pub struct AdvancedDecoderLayerPlan {
  priv layer_index : Int
  priv attention : AdvancedAttentionPlan
} derive(Debug, Eq)

///|
/// Optional DSpark prediction module attached outside the base decoder layers.
pub struct DsparkExecutionPlan {
  priv block_size : Int
  priv noise_token_id : Int
  priv target_layer_ids : ReadOnlyArray[Int]
  priv markov_rank : Int
} derive(Debug, Eq)

///|
/// Auxiliary prediction work executed after the base decoder graph.
pub(all) enum AuxiliaryPredictionPlan {
  MultiTokenPrediction(layer_count~ : Int)
  MultiTokenPredictionWithDspark(
    layer_count~ : Int,
    dspark~ : DsparkExecutionPlan
  )
} derive(Debug, Eq)

///|
/// Validated immutable semantic execution plan for advanced decoder-only
/// architectures. It selects no device, artifact, kernel, or scheduling
/// policy.
pub struct AdvancedDecoderExecutionPlan {
  priv schema_version : Int
  priv profile_identity : String
  priv content_digest : @spec.ContentDigest
  priv shape : AdvancedDecoderShape
  priv low_rank_attention : LowRankAttentionPlan
  priv position_encoding : PositionEncodingPlan
  priv hyper_connection : HyperConnectionPlan
  priv moe : MoeExecutionPlan
  priv layers : ReadOnlyArray[AdvancedDecoderLayerPlan]
  priv auxiliary_prediction : AuxiliaryPredictionPlan
  priv plan_digest : @spec.PlanDigest
} derive(Debug, Eq)
