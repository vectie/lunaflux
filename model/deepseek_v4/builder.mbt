///|
/// Exact attention semantic selected by one DeepSeek-V4 decoder layer.
pub(all) enum DeepSeekV4AttentionMode {
  SlidingWindowCausal
  CompressedSparse
  HeavilyCompressed
} derive(Debug, Eq)

///|
/// Immutable startup semantic plan for one exact DeepSeek-V4 checkpoint.
///
/// This is deliberately not `model/plan.ModelPlan`: the current generic plan
/// vocabulary cannot express mHC, the CSA/HCA hybrid, or routed experts. A
/// backend must first satisfy `requirements`; no dense-Llama substitution is
/// permitted.
pub struct DeepSeekV4SemanticPlan {
  priv metadata : @deepseek_v4_spec.DeepSeekV4ModelMetadata
  priv requirements : @family_contract.ModelFamilyRequirements
  priv execution_plan : @advanced_decoder_execution_plan.AdvancedDecoderExecutionPlan
} derive(Debug, Eq)

///|
/// Complete startup capabilities required by every schema-v1 DeepSeek-V4
/// profile. These are semantic requirements, never request-path dispatch tags.
pub fn requirements() -> @family_contract.ModelFamilyRequirements {
  try! @family_contract.ModelFamilyRequirements::new(
    family=DeepSeekV4,
    architecture="DeepseekV4ForCausalLM",
    model_type="deepseek_v4",
    schema_version=1,
    workload=DecoderOnlyText,
    capabilities=[
      DenseCausalAttention,
      QueryKeyRmsNorm,
      MixtureOfExperts,
      MultiTokenPrediction,
      HyperConnections,
      CompressedSparseAttention,
      HeavilyCompressedAttention,
    ],
  )
}

///|
/// Constructs an immutable family semantic plan. Construction does not claim
/// that the current generic executor implements the required capabilities.
pub fn build(
  metadata : @deepseek_v4_spec.DeepSeekV4ModelMetadata,
) -> DeepSeekV4SemanticPlan {
  {
    metadata,
    requirements: requirements(),
    execution_plan: build_advanced_execution_plan(metadata),
  }
}

///|
fn build_decoder_layers(
  spec : @deepseek_v4_spec.DeepSeekV4ModelSpec,
) -> Array[@advanced_decoder_execution_plan.AdvancedDecoderLayerPlan] {
  let layers : Array[@advanced_decoder_execution_plan.AdvancedDecoderLayerPlan] = []
  for layer in 0..<spec.layer_count() {
    let attention = match spec.compression_ratio(layer) {
      Some(0) =>
        try! @advanced_decoder_execution_plan.AdvancedAttentionPlan::sliding_window_causal(
          spec.sliding_window(),
        )
      Some(4) =>
        try! @advanced_decoder_execution_plan.AdvancedAttentionPlan::indexed_compressed(
          compression_ratio=4,
          index_head_count=spec.index_head_count(),
          index_head_dimension=spec.index_head_dimension(),
          index_top_k=spec.index_top_k(),
        )
      Some(ratio) =>
        try! @advanced_decoder_execution_plan.AdvancedAttentionPlan::deterministic_compressed(
          compression_ratio=ratio,
          window_size=spec.sliding_window(),
        )
      None => abort("closed DeepSeek V4 profile lost a base attention layer")
    }
    layers.push(
      try! @advanced_decoder_execution_plan.AdvancedDecoderLayerPlan::new(
        layer_index=layer,
        attention~,
      ),
    )
  }
  layers
}

///|
fn build_auxiliary_prediction(
  spec : @deepseek_v4_spec.DeepSeekV4ModelSpec,
) -> @advanced_decoder_execution_plan.AuxiliaryPredictionPlan {
  if spec.has_dspark() {
    let dspark = try! @advanced_decoder_execution_plan.DsparkExecutionPlan::new(
      block_size=spec.dspark_block_size(),
      noise_token_id=spec.dspark_noise_token_id(),
      target_layer_ids=spec.dspark_target_layer_ids(),
      markov_rank=spec.dspark_markov_rank(),
      decoder_layer_count=spec.layer_count(),
    )
    try! @advanced_decoder_execution_plan.AuxiliaryPredictionPlan::multi_token_prediction_with_dspark(
      layer_count=spec.prediction_stage_count(),
      dspark~,
    )
  } else {
    try! @advanced_decoder_execution_plan.AuxiliaryPredictionPlan::multi_token_prediction(
      spec.next_token_prediction_layer_count(),
    )
  }
}

///|
/// Projects the exact DeepSeek semantic plan into the family-neutral workload
/// envelope used by startup admission. Content and canonical execution-plan
/// identity are joined before the projection is returned.
pub fn DeepSeekV4SemanticPlan::workload_execution_plan(
  self : Self,
) -> @workload_execution_plan.WorkloadExecutionPlan {
  let execution = self.execution_plan
  let identity = try! @spec.ModelIdentity::new(
    execution.content_digest(),
    execution.plan_digest(),
  )
  try! @workload_execution_plan.WorkloadExecutionPlan::decoder_only(
    identity~,
    schema_version=execution.schema_version(),
    decoder_layer_count=execution.shape().decoder_layer_count(),
    maximum_token_context=execution.shape().maximum_position_count(),
  )
}

///|
fn build_advanced_execution_plan(
  metadata : @deepseek_v4_spec.DeepSeekV4ModelMetadata,
) -> @advanced_decoder_execution_plan.AdvancedDecoderExecutionPlan {
  let spec = metadata.spec()
  let shape = try! @advanced_decoder_execution_plan.AdvancedDecoderShape::new(
    vocabulary_size=spec.vocabulary_size(),
    hidden_size=spec.hidden_size(),
    decoder_layer_count=spec.layer_count(),
    attention_head_count=spec.attention_head_count(),
    kv_head_count=spec.kv_head_count(),
    attention_head_dimension=spec.head_dimension(),
    rope_head_dimension=spec.rope_head_dimension(),
    query_low_rank=spec.q_lora_rank(),
    output_low_rank=spec.o_lora_rank(),
    output_group_count=spec.output_group_count(),
    maximum_position_count=spec.max_position_embeddings(),
  )
  let low_rank_attention = try! @advanced_decoder_execution_plan.LowRankAttentionPlan::new(
    query_rank=spec.q_lora_rank(),
    output_rank=spec.o_lora_rank(),
    output_group_count=spec.output_group_count(),
    query_key_norm_epsilon=spec.rms_norm_epsilon(),
  )
  let position_encoding = try! @advanced_decoder_execution_plan.PositionEncodingPlan::new(
    rope_theta=spec.rope_theta(),
    compressed_rope_theta=spec.compressed_rope_theta(),
    original_position_count=spec.original_position_embeddings(),
    scaling_factor=spec.rope_scaling_factor(),
    beta_fast=spec.rope_beta_fast(),
    beta_slow=spec.rope_beta_slow(),
  )
  let hyper_connection = try! @advanced_decoder_execution_plan.HyperConnectionPlan::new(
    width=spec.hyper_connection_width(),
    control_epsilon=spec.hyper_connection_epsilon(),
    normalization_epsilon=spec.rms_norm_epsilon(),
    sinkhorn_iterations=spec.hyper_connection_sinkhorn_iterations(),
  )
  let moe = try! @advanced_decoder_execution_plan.MoeExecutionPlan::new(
    routed_expert_count=spec.routed_expert_count(),
    shared_expert_count=spec.shared_expert_count(),
    experts_per_token=spec.experts_per_token(),
    intermediate_size=spec.moe_intermediate_size(),
    routed_scaling_factor=spec.routed_scaling_factor(),
    normalize_selected_probabilities=spec.normalizes_selected_expert_probabilities(),
    score_function=SqrtSoftplus,
    selection_method=TokenHashThenNoAuxiliaryLossTokenChoice(
      hash_layer_count=spec.hash_layer_count(),
    ),
    activation=ClampedSwiGlu(spec.swiglu_limit()),
  )
  try! @advanced_decoder_execution_plan.AdvancedDecoderExecutionPlan::new(
    schema_version=1,
    profile_identity=spec.execution_profile_identity(),
    content_digest=metadata.content_digest(),
    shape~,
    low_rank_attention~,
    position_encoding~,
    hyper_connection~,
    moe~,
    layers=build_decoder_layers(spec),
    auxiliary_prediction=build_auxiliary_prediction(spec),
  )
}

///|
pub fn DeepSeekV4SemanticPlan::metadata(
  self : Self,
) -> @deepseek_v4_spec.DeepSeekV4ModelMetadata {
  self.metadata
}

///|
pub fn DeepSeekV4SemanticPlan::content_digest(
  self : Self,
) -> @spec.ContentDigest {
  self.metadata.content_digest()
}

///|
pub fn DeepSeekV4SemanticPlan::requirements(
  self : Self,
) -> @family_contract.ModelFamilyRequirements {
  self.requirements
}

///|
/// Returns the complete architecture-neutral advanced decoder execution plan.
/// Downstream backends must bind its exact digest before selecting artifacts.
pub fn DeepSeekV4SemanticPlan::execution_plan(
  self : Self,
) -> @advanced_decoder_execution_plan.AdvancedDecoderExecutionPlan {
  self.execution_plan
}

///|
/// Joins this exact semantic profile with a backend capability set. A partial
/// backend reports every missing capability in stable declared order. Success
/// grants no artifact, device, runtime, or physical execution authority.
pub fn DeepSeekV4SemanticPlan::admit_backend(
  self : Self,
  backend : @family_contract.ModelBackendCapabilities,
) -> @family_contract.ModelFamilyAdmission {
  self.requirements.admit(backend)
}

///|
/// Returns the exact hybrid-attention mode for one base decoder layer.
/// Attached MTP/DSpark attention profiles are not silently treated as base
/// layers and therefore remain outside this accessor.
pub fn DeepSeekV4SemanticPlan::attention_mode(
  self : Self,
  layer : Int,
) -> DeepSeekV4AttentionMode? {
  let spec = self.metadata.spec()
  if layer < 0 || layer >= spec.layer_count() {
    return None
  }
  match spec.compression_ratio(layer) {
    Some(0) => Some(SlidingWindowCausal)
    Some(4) => Some(CompressedSparse)
    Some(128) => Some(HeavilyCompressed)
    _ => None
  }
}
