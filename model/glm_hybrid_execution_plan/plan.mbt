///|
fn valid_identity_component(value : StringView) -> Bool {
  !value.is_empty() &&
  value.length() <= 128 &&
  value.all(c => {
    c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' | '.' | '/' | ':')
  })
}

///|
fn validate_auxiliary(
  auxiliary : @advanced_decoder_execution_plan.AuxiliaryPredictionPlan,
) -> Unit raise GlmHybridExecutionPlanError {
  match auxiliary {
    MultiTokenPrediction(layer_count~) =>
      if layer_count <= 0 {
        raise InvalidAuxiliaryPrediction
      }
    MultiTokenPredictionWithDspark(layer_count~, dspark=_) =>
      if layer_count <= 0 {
        raise InvalidAuxiliaryPrediction
      }
  }
}

///|
fn validate_layer_schedule(
  layers : ArrayView[GlmHybridLayerPlan],
  expected_count : Int,
  kda : GlmKdaExecutionPlan?,
) -> Unit raise GlmHybridExecutionPlanError {
  if layers.length() != expected_count {
    raise LayerCountMismatch
  }
  let mut has_sparse_index = false
  let mut has_kda_layer = false
  let mut has_dense_layer = false
  let mut has_moe_layer = false
  for index, layer in layers {
    if layer.layer_index() != index {
      raise InvalidLayerOrder
    }
    match layer.attention() {
      DynamicSparse(ComputeTopK) => has_sparse_index = true
      DynamicSparse(ReusePreviousTopK) =>
        if !has_sparse_index {
          raise InvalidSparseIndexSchedule
        }
      KimiDelta => has_kda_layer = true
    }
    match layer.feed_forward() {
      DenseSwiGlu => has_dense_layer = true
      RoutedMoe => has_moe_layer = true
    }
  }
  if has_kda_layer && kda is None {
    raise MissingKdaContract
  }
  if !has_kda_layer && kda is Some(_) {
    raise UnusedKdaContract
  }
  if !has_dense_layer {
    raise MissingDenseLayer
  }
  if !has_moe_layer {
    raise MissingRoutedMoeLayer
  }
}

///|
/// Validates and seals a complete GLM hybrid execution plan.
pub fn GlmHybridExecutionPlan::new(
  content_digest~ : @spec.ContentDigest,
  schema_version~ : Int,
  profile_identity~ : String,
  architecture~ : String,
  model_type~ : String,
  vocabulary_size~ : Int,
  hidden_size~ : Int,
  dense_intermediate_size~ : Int,
  decoder_layer_count~ : Int,
  context_length~ : Int,
  sparse_attention~ : GlmDynamicSparseAttentionPlan,
  kda~ : GlmKdaExecutionPlan?,
  hyper_connection~ : @advanced_decoder_execution_plan.HyperConnectionPlan?,
  moe~ : @advanced_decoder_execution_plan.MoeExecutionPlan,
  moe_router_float32~ : Bool,
  moe_correction_bias~ : Bool,
  layers~ : ArrayView[GlmHybridLayerPlan],
  auxiliary_prediction~ : @advanced_decoder_execution_plan.AuxiliaryPredictionPlan,
  vision_conditioning~ : GlmVisionConditioningPlan?,
) -> GlmHybridExecutionPlan raise GlmHybridExecutionPlanError {
  if schema_version != 1 {
    raise InvalidSchemaVersion
  }
  if !valid_identity_component(profile_identity) ||
    !valid_identity_component(architecture) ||
    !valid_identity_component(model_type) {
    raise InvalidProfileIdentity
  }
  if vocabulary_size <= 0 ||
    hidden_size <= 0 ||
    dense_intermediate_size <= 0 ||
    decoder_layer_count <= 0 ||
    context_length <= 0 ||
    layers.is_empty() ||
    sparse_attention.index_topk() > context_length {
    raise InvalidModelGeometry
  }
  validate_layer_schedule(layers, decoder_layer_count, kda)
  validate_auxiliary(auxiliary_prediction)
  match vision_conditioning {
    Some(vision) =>
      if vision.decoder_projection_size() != hidden_size {
        raise InvalidVisionConditioning
      }
    None => ()
  }
  let plan_digest = canonical_plan_digest(
    content_digest, schema_version, profile_identity, architecture, model_type, vocabulary_size,
    hidden_size, dense_intermediate_size, decoder_layer_count, context_length, sparse_attention,
    kda, hyper_connection, moe, moe_router_float32, moe_correction_bias, layers,
    auxiliary_prediction, vision_conditioning,
  )
  let identity = try! @spec.ModelIdentity::new(content_digest, plan_digest)
  {
    identity,
    schema_version,
    profile_identity,
    architecture,
    model_type,
    vocabulary_size,
    hidden_size,
    dense_intermediate_size,
    decoder_layer_count,
    context_length,
    sparse_attention,
    kda,
    hyper_connection,
    moe,
    moe_router_float32,
    moe_correction_bias,
    layers: ReadOnlyArray::from_array(layers),
    auxiliary_prediction,
    vision_conditioning,
  }
}
