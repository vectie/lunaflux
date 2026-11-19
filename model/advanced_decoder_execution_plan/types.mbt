///|
/// Immutable architecture geometry shared by all decoder layers.
pub struct AdvancedDecoderShape {
  priv vocabulary_size : Int32
  priv hidden_size : Int32
  priv decoder_layer_count : Int32
  priv attention_head_count : Int32
  priv kv_head_count : Int32
  priv attention_head_dimension : Int32
  priv rope_head_dimension : Int32
  priv query_low_rank : Int32
  priv output_low_rank : Int32
  priv output_group_count : Int32
  priv maximum_position_count : Int32
} derive(Debug, Eq)

///|
/// Low-rank attention projection and normalization semantics.
pub struct LowRankAttentionPlan {
  priv query_rank : Int32
  priv output_rank : Int32
  priv output_group_count : Int32
  priv query_key_norm_epsilon : Double
} derive(Debug, Eq)

///|
/// Architecture-neutral position encoding with a distinct theta for indexed
/// compressed attention.
pub struct PositionEncodingPlan {
  priv rope_theta : Double
  priv compressed_rope_theta : Double
  priv original_position_count : Int32
  priv scaling_factor : Double
  priv beta_fast : Double
  priv beta_slow : Double
} derive(Debug, Eq)

///|
/// Multi-stream residual mixing applied around every decoder block.
pub struct HyperConnectionPlan {
  priv width : Int32
  priv control_epsilon : Double
  priv normalization_epsilon : Double
  priv sinkhorn_iterations : Int32
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
  TokenHashThenNoAuxiliaryLossTokenChoice(hash_layer_count~ : Int32)
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
  priv routed_expert_count : Int32
  priv shared_expert_count : Int32
  priv experts_per_token : Int32
  priv intermediate_size : Int32
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
  SlidingWindowCausal(window_size~ : Int32)
  IndexedCompressed(
    compression_ratio~ : Int32,
    index_head_count~ : Int32,
    index_head_dimension~ : Int32,
    index_top_k~ : Int32
  )
  DeterministicCompressed(compression_ratio~ : Int32, window_size~ : Int32)
} derive(Debug, Eq)

///|
/// One decoder layer in exact execution order.
pub struct AdvancedDecoderLayerPlan {
  priv layer_index : Int32
  priv attention : AdvancedAttentionPlan
} derive(Debug, Eq)

///|
/// Optional DSpark prediction module attached outside the base decoder layers.
pub struct DsparkExecutionPlan {
  priv block_size : Int32
  priv noise_token_id : Int32
  priv target_layer_ids : ReadOnlyArray[Int32]
  priv markov_rank : Int32
} derive(Debug, Eq)

///|
/// Auxiliary prediction work executed after the base decoder graph.
pub(all) enum AuxiliaryPredictionPlan {
  MultiTokenPrediction(layer_count~ : Int32)
  MultiTokenPredictionWithDspark(
    layer_count~ : Int32,
    dspark~ : DsparkExecutionPlan
  )
} derive(Debug, Eq)

///|
/// Validated immutable semantic execution plan for advanced decoder-only
/// architectures. It selects no device, artifact, kernel, or scheduling
/// policy.
pub struct AdvancedDecoderExecutionPlan {
  priv schema_version : Int32
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

///|
pub extend AdvancedDecoderShape with @debug.Debug::{to_repr}

///|
pub extend AdvancedDecoderShape with Eq::{not_equal, equal}

///|
pub extend LowRankAttentionPlan with @debug.Debug::{to_repr}

///|
pub extend LowRankAttentionPlan with Eq::{not_equal, equal}

///|
pub extend PositionEncodingPlan with @debug.Debug::{to_repr}

///|
pub extend PositionEncodingPlan with Eq::{not_equal, equal}

///|
pub extend HyperConnectionPlan with @debug.Debug::{to_repr}

///|
pub extend HyperConnectionPlan with Eq::{not_equal, equal}

///|
pub extend ExpertScoreFunction with @debug.Debug::{to_repr}

///|
pub extend ExpertScoreFunction with Eq::{not_equal, equal}

///|
pub extend ExpertSelectionMethod with @debug.Debug::{to_repr}

///|
pub extend ExpertSelectionMethod with Eq::{not_equal, equal}

///|
pub extend ExpertActivation with @debug.Debug::{to_repr}

///|
pub extend ExpertActivation with Eq::{not_equal, equal}

///|
pub extend MoeExecutionPlan with @debug.Debug::{to_repr}

///|
pub extend MoeExecutionPlan with Eq::{not_equal, equal}

///|
pub extend AdvancedAttentionPlan with @debug.Debug::{to_repr}

///|
pub extend AdvancedAttentionPlan with Eq::{not_equal, equal}

///|
pub extend AdvancedDecoderLayerPlan with @debug.Debug::{to_repr}

///|
pub extend AdvancedDecoderLayerPlan with Eq::{not_equal, equal}

///|
pub extend DsparkExecutionPlan with @debug.Debug::{to_repr}

///|
pub extend DsparkExecutionPlan with Eq::{not_equal, equal}

///|
pub extend AuxiliaryPredictionPlan with @debug.Debug::{to_repr}

///|
pub extend AuxiliaryPredictionPlan with Eq::{not_equal, equal}

///|
pub extend AdvancedDecoderExecutionPlan with @debug.Debug::{to_repr}

///|
pub extend AdvancedDecoderExecutionPlan with Eq::{not_equal, equal}
