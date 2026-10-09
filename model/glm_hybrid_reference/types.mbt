///|
pub(all) suberror GlmHybridReferenceError {
  InvalidLimits
  BoundsExceeded
  InvalidShape
  NonFiniteInput
  InvalidProbability
  InvalidLayerOrder
  MissingDsaState
  StaleDsaState
  DuplicateIndex
  IndexOutOfBounds
  InvalidVisionGeometry
  SharedPrimitiveFailure
} derive(Debug, Eq)

///|
/// Explicit tiny-fixture bounds. Reference execution is not a production
/// fallback and cannot be configured to model full checkpoint dimensions.
pub struct GlmHybridReferenceLimits {
  priv maximum_sequence_length : Int
  priv maximum_head_count : Int
  priv maximum_head_dimension : Int
  priv maximum_channel_count : Int
  priv maximum_vision_patch_count : Int
}

///|
pub fn GlmHybridReferenceLimits::new(
  maximum_sequence_length~ : Int,
  maximum_head_count~ : Int,
  maximum_head_dimension~ : Int,
  maximum_channel_count~ : Int,
  maximum_vision_patch_count~ : Int,
) -> GlmHybridReferenceLimits raise GlmHybridReferenceError {
  if maximum_sequence_length <= 0 ||
    maximum_sequence_length > 256 ||
    maximum_head_count <= 0 ||
    maximum_head_count > 128 ||
    maximum_head_dimension <= 0 ||
    maximum_head_dimension > 256 ||
    maximum_channel_count <= 0 ||
    maximum_channel_count > 32768 ||
    maximum_vision_patch_count <= 0 {
    raise InvalidLimits
  }
  if maximum_vision_patch_count > 4096 {
    raise InvalidLimits
  }
  {
    maximum_sequence_length,
    maximum_head_count,
    maximum_head_dimension,
    maximum_channel_count,
    maximum_vision_patch_count,
  }
}

///|
pub struct GlmDsaPooledKeys {
  priv pool_count : Int
  priv pool_size : Int
  priv head_dimension : Int
  priv keys : ReadOnlyArray[Double]
  priv raw_indices : ReadOnlyArray[Int]
  priv valid : ReadOnlyArray[Bool]
}

///|
pub struct GlmDsaTopKState {
  priv source_layer : Int
  priv last_consumer_layer : Int
  priv key_count : Int
  priv indices : ReadOnlyArray[Int]
}

///|
pub struct GlmKdaConvolutionResult {
  priv sequence_length : Int
  priv channel_count : Int
  priv output : ReadOnlyArray[Double]
  priv final_state : ReadOnlyArray[Double]
}

///|
pub struct GlmKdaProjectionResult {
  priv row_count : Int
  priv projected_width : Int
  priv control_rank : Int
  priv head_count : Int
  priv query : ReadOnlyArray[Double]
  priv key : ReadOnlyArray[Double]
  priv value : ReadOnlyArray[Double]
  priv forget_low_rank : ReadOnlyArray[Double]
  priv raw_forget : ReadOnlyArray[Double]
  priv beta : ReadOnlyArray[Double]
}

///|
pub struct GlmKdaOutputGateResult {
  priv low_rank : ReadOnlyArray[Double]
  priv gate : ReadOnlyArray[Double]
}

///|
pub struct GlmKdaGatedNormResult {
  priv row_count : Int
  priv head_count : Int
  priv head_dimension : Int
  priv values : ReadOnlyArray[Double]
}

///|
pub struct GlmKdaOutputProjectionResult {
  priv row_count : Int
  priv input_width : Int
  priv output_width : Int
  priv values : ReadOnlyArray[Double]
}

///|
pub struct GlmKdaDecayControlResult {
  priv row_count : Int
  priv head_count : Int
  priv head_dimension : Int
  priv log_decay : ReadOnlyArray[Double]
}

///|
pub struct GlmMoeRouterProjectionResult {
  priv row_count : Int
  priv hidden_width : Int
  priv expert_count : Int
  priv logits : ReadOnlyArray[Double]
}

///|
pub struct GlmHyperConnectionFunctionResult {
  priv row_count : Int
  priv stream_count : Int
  priv hidden_width : Int
  priv post : ReadOnlyArray[Double]
  priv initial_combination : ReadOnlyArray[Double]
  priv collapsed : ReadOnlyArray[Double]
}

///|
pub struct GlmHyperConnectionSinkhornResult {
  priv row_count : Int
  priv stream_count : Int
  priv combination : ReadOnlyArray[Double]
}

///|
pub struct GlmMoeSigmoidCorrectionResult {
  priv row_count : Int
  priv expert_count : Int
  priv scores : ReadOnlyArray[Float]
  priv scores_for_choice : ReadOnlyArray[Float]
}

///|
pub struct GlmMoeGroupedRoutingResult {
  priv row_count : Int
  priv experts_per_token : Int
  priv selected_indices : ReadOnlyArray[Int]
  priv selected_weights : ReadOnlyArray[Float]
}

///|
pub struct GlmMoeFeedForwardResult {
  priv row_count : Int
  priv hidden_width : Int
  priv values : ReadOnlyArray[Double]
}

///|
pub struct GlmResidualCompositionResult {
  priv row_count : Int
  priv stream_count : Int
  priv hidden_width : Int
  priv values : ReadOnlyArray[Double]
}

///|
pub struct GlmDsaKvAssemblyResult {
  priv row_count : Int
  priv head_count : Int
  priv key_width : Int
  priv value_width : Int
  priv keys : ReadOnlyArray[Double]
  priv values : ReadOnlyArray[Double]
}

///|
pub struct GlmKdaRecurrentResult {
  priv sequence_length : Int
  priv head_count : Int
  priv head_dimension : Int
  priv output : ReadOnlyArray[Double]
  priv final_state : ReadOnlyArray[Double]
}

///|
pub struct GlmVisionReferenceResult {
  priv row_count : Int
  priv width : Int
  priv values : ReadOnlyArray[Double]
}

///|
pub(all) enum GlmReferenceAttentionStep {
  DsaCompute
  DsaReuse(source_layer~ : Int)
  Kda
} derive(Debug, Eq)

///|
pub(all) enum GlmReferenceFeedForwardStep {
  Dense
  Moe
} derive(Debug, Eq)

///|
pub struct GlmReferenceLayerStep {
  priv layer_index : Int
  priv attention : GlmReferenceAttentionStep
  priv feed_forward : GlmReferenceFeedForwardStep
  priv hyper_connection : Bool
}

///|
pub struct GlmHybridReferenceSchedule {
  priv steps : ReadOnlyArray[GlmReferenceLayerStep]
  priv dsa_compute_count : Int
  priv dsa_reuse_count : Int
  priv kda_count : Int
  priv dense_count : Int
  priv moe_count : Int
  priv hyper_connection_site_count : Int
  priv vision_layer_count : Int
}

///|
/// Tiny deterministic output proving the GLM adapter dispatches shared MoE
/// routing/combination and optional mHC Sinkhorn/mixing primitives.
pub struct GlmSharedAdvancedFixture {
  priv moe_output : Float
  priv mixed_hyper_streams : ReadOnlyArray[Float]
}
