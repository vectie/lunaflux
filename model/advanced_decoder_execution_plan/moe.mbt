///|
pub fn MoeExecutionPlan::new(
  routed_expert_count~ : Int,
  shared_expert_count~ : Int,
  experts_per_token~ : Int,
  intermediate_size~ : Int,
  routed_scaling_factor~ : Double,
  normalize_selected_probabilities~ : Bool,
  score_function~ : ExpertScoreFunction,
  selection_method~ : ExpertSelectionMethod,
  activation~ : ExpertActivation,
) -> MoeExecutionPlan raise AdvancedDecoderPlanError {
  let activation_valid = match activation {
    SwiGlu => true
    ClampedSwiGlu(limit) => finite_positive(limit)
  }
  let selection_valid = match selection_method {
    NoAuxiliaryLossTokenChoice => true
    TokenHashThenNoAuxiliaryLossTokenChoice(hash_layer_count~) =>
      hash_layer_count > 0
  }
  if routed_expert_count <= 0 ||
    shared_expert_count < 0 ||
    experts_per_token <= 0 ||
    experts_per_token > routed_expert_count ||
    intermediate_size <= 0 ||
    !finite_positive(routed_scaling_factor) ||
    !activation_valid ||
    !selection_valid {
    raise InvalidExpertRouting
  }
  {
    routed_expert_count,
    shared_expert_count,
    experts_per_token,
    intermediate_size,
    routed_scaling_factor,
    normalize_selected_probabilities,
    score_function,
    selection_method,
    activation,
  }
}

///|
pub fn MoeExecutionPlan::hash_routed_layer_count(self : Self) -> Int {
  match self.selection_method {
    NoAuxiliaryLossTokenChoice => 0
    TokenHashThenNoAuxiliaryLossTokenChoice(hash_layer_count~) =>
      hash_layer_count
  }
}

///|
pub fn AdvancedAttentionPlan::sliding_window_causal(
  window_size : Int,
) -> AdvancedAttentionPlan raise AdvancedDecoderPlanError {
  if window_size <= 0 {
    raise InvalidLayerAttention
  }
  SlidingWindowCausal(window_size~)
}

///|
pub fn AdvancedAttentionPlan::indexed_compressed(
  compression_ratio~ : Int,
  index_head_count~ : Int,
  index_head_dimension~ : Int,
  index_top_k~ : Int,
) -> AdvancedAttentionPlan raise AdvancedDecoderPlanError {
  if compression_ratio <= 1 ||
    index_head_count <= 0 ||
    index_head_dimension <= 0 ||
    index_top_k <= 0 {
    raise InvalidLayerAttention
  }
  IndexedCompressed(
    compression_ratio~,
    index_head_count~,
    index_head_dimension~,
    index_top_k~,
  )
}

///|
pub fn AdvancedAttentionPlan::deterministic_compressed(
  compression_ratio~ : Int,
  window_size~ : Int,
) -> AdvancedAttentionPlan raise AdvancedDecoderPlanError {
  if compression_ratio <= 1 || window_size <= 0 {
    raise InvalidLayerAttention
  }
  DeterministicCompressed(compression_ratio~, window_size~)
}

///|
pub fn AdvancedDecoderLayerPlan::new(
  layer_index~ : Int,
  attention~ : AdvancedAttentionPlan,
) -> AdvancedDecoderLayerPlan raise AdvancedDecoderPlanError {
  if layer_index < 0 {
    raise InvalidLayerOrder
  }
  match attention {
    SlidingWindowCausal(window_size~) if window_size <= 0 =>
      raise InvalidLayerAttention
    IndexedCompressed(
      compression_ratio~,
      index_head_count~,
      index_head_dimension~,
      index_top_k~
    ) if compression_ratio <= 1 ||
      index_head_count <= 0 ||
      index_head_dimension <= 0 ||
      index_top_k <= 0 => raise InvalidLayerAttention
    DeterministicCompressed(compression_ratio~, window_size~) if compression_ratio <=
      1 ||
      window_size <= 0 => raise InvalidLayerAttention
    _ => ()
  }
  { layer_index, attention, }
}

///|
pub fn DsparkExecutionPlan::new(
  block_size~ : Int,
  noise_token_id~ : Int,
  target_layer_ids~ : ArrayView[Int],
  markov_rank~ : Int,
  decoder_layer_count~ : Int,
) -> DsparkExecutionPlan raise AdvancedDecoderPlanError {
  if block_size <= 0 ||
    noise_token_id < 0 ||
    target_layer_ids.is_empty() ||
    markov_rank <= 0 ||
    decoder_layer_count <= 0 {
    raise InvalidAuxiliaryPrediction
  }
  let mut previous = -1
  for layer in target_layer_ids {
    if layer <= previous || layer < 0 || layer >= decoder_layer_count {
      raise InvalidAuxiliaryPrediction
    }
    previous = layer
  }
  {
    block_size,
    noise_token_id,
    target_layer_ids: ReadOnlyArray::from_array(target_layer_ids),
    markov_rank,
  }
}

///|
pub fn AuxiliaryPredictionPlan::multi_token_prediction(
  layer_count : Int,
) -> AuxiliaryPredictionPlan raise AdvancedDecoderPlanError {
  if layer_count <= 0 {
    raise InvalidAuxiliaryPrediction
  }
  MultiTokenPrediction(layer_count~)
}

///|
pub fn AuxiliaryPredictionPlan::multi_token_prediction_with_dspark(
  layer_count~ : Int,
  dspark~ : DsparkExecutionPlan,
) -> AuxiliaryPredictionPlan raise AdvancedDecoderPlanError {
  if layer_count <= 0 {
    raise InvalidAuxiliaryPrediction
  }
  MultiTokenPredictionWithDspark(layer_count~, dspark~)
}
