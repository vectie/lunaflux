///|
fn valid_profile_identity(value : StringView) -> Bool {
  value.length() > 0 &&
  value.length() <= 128 &&
  value.all(c => {
    c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' | '/' | ':')
  })
}

///|
fn validate_auxiliary_prediction(
  auxiliary : AuxiliaryPredictionPlan,
) -> Unit raise AdvancedDecoderPlanError {
  match auxiliary {
    MultiTokenPrediction(layer_count~)
    | MultiTokenPredictionWithDspark(layer_count~, ..) =>
      if layer_count <= 0 {
        raise InvalidAuxiliaryPrediction
      }
  }
}

///|
pub fn AdvancedDecoderExecutionPlan::new(
  schema_version~ : Int,
  profile_identity~ : String,
  content_digest~ : @spec.ContentDigest,
  shape~ : AdvancedDecoderShape,
  low_rank_attention~ : LowRankAttentionPlan,
  position_encoding~ : PositionEncodingPlan,
  hyper_connection~ : HyperConnectionPlan,
  moe~ : MoeExecutionPlan,
  layers~ : ArrayView[AdvancedDecoderLayerPlan],
  auxiliary_prediction~ : AuxiliaryPredictionPlan,
) -> AdvancedDecoderExecutionPlan raise AdvancedDecoderPlanError {
  if schema_version <= 0 {
    raise InvalidSchemaVersion
  }
  if !valid_profile_identity(profile_identity) {
    raise InvalidProfileIdentity
  }
  if low_rank_attention.query_rank() != shape.query_low_rank() ||
    low_rank_attention.output_rank() != shape.output_low_rank() ||
    low_rank_attention.output_group_count() != shape.output_group_count() {
    raise InvalidAttentionGeometry
  }
  if position_encoding.original_position_count() >
    shape.maximum_position_count() {
    raise InvalidPositionEncoding
  }
  if moe.hash_routed_layer_count() > shape.decoder_layer_count() {
    raise InvalidExpertRouting
  }
  if layers.length() != shape.decoder_layer_count() {
    raise LayerCountMismatch
  }
  validate_auxiliary_prediction(auxiliary_prediction)
  for index, layer in layers {
    if layer.layer_index() != index {
      raise InvalidLayerOrder
    }
    match layer.attention() {
      SlidingWindowCausal(window_size~) =>
        if window_size > shape.maximum_position_count() {
          raise InvalidLayerAttention
        }
      IndexedCompressed(
        compression_ratio=_,
        index_head_count~,
        index_head_dimension~,
        index_top_k~
      ) =>
        if index_head_count > shape.attention_head_count() ||
          shape.attention_head_count() % index_head_count != 0 ||
          index_head_dimension > shape.attention_head_dimension() ||
          index_top_k > shape.maximum_position_count() {
          raise InvalidLayerAttention
        }
      DeterministicCompressed(compression_ratio~, window_size~) =>
        if compression_ratio > shape.maximum_position_count() ||
          window_size > shape.maximum_position_count() {
          raise InvalidLayerAttention
        }
    }
  }
  let plan_digest = canonical_plan_digest(
    schema_version, profile_identity, content_digest, shape, low_rank_attention,
    position_encoding, hyper_connection, moe, layers, auxiliary_prediction,
  )
  {
    schema_version,
    profile_identity,
    content_digest,
    shape,
    low_rank_attention,
    position_encoding,
    hyper_connection,
    moe,
    layers: ReadOnlyArray::from_array(layers),
    auxiliary_prediction,
    plan_digest,
  }
}
