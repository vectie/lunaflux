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
  priv maximum_sequence_length : Int32
  priv maximum_head_count : Int32
  priv maximum_head_dimension : Int32
  priv maximum_channel_count : Int32
  priv maximum_vision_patch_count : Int32
}

///|
pub fn GlmHybridReferenceLimits::new(
  maximum_sequence_length~ : Int32,
  maximum_head_count~ : Int32,
  maximum_head_dimension~ : Int32,
  maximum_channel_count~ : Int32,
  maximum_vision_patch_count~ : Int32,
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
  priv pool_count : Int32
  priv pool_size : Int32
  priv head_dimension : Int32
  priv keys : ReadOnlyArray[Double]
  priv raw_indices : ReadOnlyArray[Int32]
  priv valid : ReadOnlyArray[Bool]
}

///|
pub struct GlmDsaTopKState {
  priv source_layer : Int32
  priv last_consumer_layer : Int32
  priv key_count : Int32
  priv indices : ReadOnlyArray[Int32]
}

///|
pub struct GlmKdaConvolutionResult {
  priv sequence_length : Int32
  priv channel_count : Int32
  priv output : ReadOnlyArray[Double]
  priv final_state : ReadOnlyArray[Double]
}

///|
pub struct GlmKdaProjectionResult {
  priv row_count : Int32
  priv projected_width : Int32
  priv control_rank : Int32
  priv head_count : Int32
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
  priv row_count : Int32
  priv head_count : Int32
  priv head_dimension : Int32
  priv values : ReadOnlyArray[Double]
}

///|
pub struct GlmKdaOutputProjectionResult {
  priv row_count : Int32
  priv input_width : Int32
  priv output_width : Int32
  priv values : ReadOnlyArray[Double]
}

///|
pub struct GlmKdaDecayControlResult {
  priv row_count : Int32
  priv head_count : Int32
  priv head_dimension : Int32
  priv log_decay : ReadOnlyArray[Double]
}

///|
pub struct GlmMoeRouterProjectionResult {
  priv row_count : Int32
  priv hidden_width : Int32
  priv expert_count : Int32
  priv logits : ReadOnlyArray[Double]
}

///|
pub struct GlmHyperConnectionFunctionResult {
  priv row_count : Int32
  priv stream_count : Int32
  priv hidden_width : Int32
  priv post : ReadOnlyArray[Double]
  priv initial_combination : ReadOnlyArray[Double]
  priv collapsed : ReadOnlyArray[Double]
}

///|
pub struct GlmHyperConnectionSinkhornResult {
  priv row_count : Int32
  priv stream_count : Int32
  priv combination : ReadOnlyArray[Double]
}

///|
pub struct GlmMoeSigmoidCorrectionResult {
  priv row_count : Int32
  priv expert_count : Int32
  priv scores : ReadOnlyArray[Float]
  priv scores_for_choice : ReadOnlyArray[Float]
}

///|
pub struct GlmMoeGroupedRoutingResult {
  priv row_count : Int32
  priv experts_per_token : Int32
  priv selected_indices : ReadOnlyArray[Int32]
  priv selected_weights : ReadOnlyArray[Float]
}

///|
pub struct GlmMoeFeedForwardResult {
  priv row_count : Int32
  priv hidden_width : Int32
  priv values : ReadOnlyArray[Double]
}

///|
pub struct GlmResidualCompositionResult {
  priv row_count : Int32
  priv stream_count : Int32
  priv hidden_width : Int32
  priv values : ReadOnlyArray[Double]
}

///|
pub struct GlmDsaKvAssemblyResult {
  priv row_count : Int32
  priv head_count : Int32
  priv key_width : Int32
  priv value_width : Int32
  priv keys : ReadOnlyArray[Double]
  priv values : ReadOnlyArray[Double]
}

///|
pub struct GlmKdaRecurrentResult {
  priv sequence_length : Int32
  priv head_count : Int32
  priv head_dimension : Int32
  priv output : ReadOnlyArray[Double]
  priv final_state : ReadOnlyArray[Double]
}

///|
pub struct GlmVisionReferenceResult {
  priv row_count : Int32
  priv width : Int32
  priv values : ReadOnlyArray[Double]
}

///|
pub(all) enum GlmReferenceAttentionStep {
  DsaCompute
  DsaReuse(source_layer~ : Int32)
  Kda
} derive(Debug, Eq)

///|
pub(all) enum GlmReferenceFeedForwardStep {
  Dense
  Moe
} derive(Debug, Eq)

///|
pub struct GlmReferenceLayerStep {
  priv layer_index : Int32
  priv attention : GlmReferenceAttentionStep
  priv feed_forward : GlmReferenceFeedForwardStep
  priv hyper_connection : Bool
}

///|
pub struct GlmHybridReferenceSchedule {
  priv steps : ReadOnlyArray[GlmReferenceLayerStep]
  priv dsa_compute_count : Int32
  priv dsa_reuse_count : Int32
  priv kda_count : Int32
  priv dense_count : Int32
  priv moe_count : Int32
  priv hyper_connection_site_count : Int32
  priv vision_layer_count : Int32
}

///|
/// Tiny deterministic output proving the GLM adapter dispatches shared MoE
/// routing/combination and optional mHC Sinkhorn/mixing primitives.
pub struct GlmSharedAdvancedFixture {
  priv moe_output : Float
  priv mixed_hyper_streams : ReadOnlyArray[Float]
}
