///|
fn full_sparse_attention(
  spec : @glm53_spec.Glm53FullSpec,
) -> @glm_hybrid_execution_plan.GlmDynamicSparseAttentionPlan {
  try! @glm_hybrid_execution_plan.GlmDynamicSparseAttentionPlan::new(
    query_head_count=spec.attention_head_count(),
    kv_head_count=spec.kv_head_count(),
    query_lora_rank=spec.q_lora_rank(),
    kv_lora_rank=spec.kv_lora_rank(),
    qk_nope_head_dimension=spec.qk_nope_head_dimension(),
    qk_rope_head_dimension=spec.qk_rope_head_dimension(),
    value_head_dimension=spec.value_head_dimension(),
    rms_norm_epsilon=spec.rms_norm_epsilon(),
    rope_theta=Some(spec.rope_theta()),
    rope_interleave=spec.rope_interleave(),
    index_head_count=spec.index_head_count(),
    index_head_dimension=spec.index_head_dimension(),
    index_topk=spec.index_topk(),
    indexer_rope_interleave=spec.indexer_rope_interleave(),
    index_share_for_mtp_iteration=spec.index_share_for_mtp_iteration(),
    index_pooling=None,
  )
}

///|
fn flash_sparse_attention(
  spec : @glm53_spec.Glm53FlashSpec,
) -> @glm_hybrid_execution_plan.GlmDynamicSparseAttentionPlan {
  let pooling = try! @glm_hybrid_execution_plan.GlmSparseIndexPoolingPlan::new(
    kernel_size=spec.index_kpool(),
    compress=spec.index_pool_compress(),
    always_select_tail=spec.index_pool_always_select_tail(),
  )
  try! @glm_hybrid_execution_plan.GlmDynamicSparseAttentionPlan::new(
    query_head_count=spec.attention_head_count(),
    kv_head_count=spec.kv_head_count(),
    query_lora_rank=spec.q_lora_rank(),
    kv_lora_rank=spec.kv_lora_rank(),
    qk_nope_head_dimension=spec.qk_nope_head_dimension(),
    qk_rope_head_dimension=spec.qk_rope_head_dimension(),
    value_head_dimension=spec.value_head_dimension(),
    rms_norm_epsilon=spec.rms_norm_epsilon(),
    rope_theta=None,
    rope_interleave=false,
    index_head_count=spec.index_head_count(),
    index_head_dimension=spec.index_head_dimension(),
    index_topk=spec.index_topk(),
    indexer_rope_interleave=spec.indexer_rope_interleave(),
    index_share_for_mtp_iteration=spec.index_share_for_mtp_iteration(),
    index_pooling=Some(pooling),
  )
}

///|
fn full_moe(
  spec : @glm53_spec.Glm53FullSpec,
) -> @advanced_decoder_execution_plan.MoeExecutionPlan {
  try! @advanced_decoder_execution_plan.MoeExecutionPlan::new(
    routed_expert_count=spec.routed_expert_count(),
    shared_expert_count=spec.shared_expert_count(),
    experts_per_token=spec.experts_per_token(),
    intermediate_size=spec.moe_intermediate_size(),
    routed_scaling_factor=spec.routed_scaling_factor(),
    normalize_selected_probabilities=spec.normalizes_selected_expert_probabilities(),
    score_function=Sigmoid,
    selection_method=NoAuxiliaryLossTokenChoice,
    activation=SwiGlu,
  )
}

///|
fn flash_moe(
  spec : @glm53_spec.Glm53FlashSpec,
) -> @advanced_decoder_execution_plan.MoeExecutionPlan {
  try! @advanced_decoder_execution_plan.MoeExecutionPlan::new(
    routed_expert_count=spec.routed_expert_count(),
    shared_expert_count=spec.shared_expert_count(),
    experts_per_token=spec.experts_per_token(),
    intermediate_size=spec.moe_intermediate_size(),
    routed_scaling_factor=spec.routed_scaling_factor(),
    normalize_selected_probabilities=spec.normalizes_selected_expert_probabilities(),
    score_function=Sigmoid,
    selection_method=NoAuxiliaryLossTokenChoice,
    activation=ClampedSwiGlu(spec.swiglu_limit()),
  )
}

///|
fn full_layers(
  spec : @glm53_spec.Glm53FullSpec,
) -> Array[@glm_hybrid_execution_plan.GlmHybridLayerPlan] {
  let layers = []
  for layer in 0..<spec.layer_count() {
    let attention : @glm_hybrid_execution_plan.GlmLayerAttentionPlan = if (try! spec.computes_index_topk(
        layer,
      )) {
      DynamicSparse(ComputeTopK)
    } else {
      DynamicSparse(ReusePreviousTopK)
    }
    let feed_forward : @glm_hybrid_execution_plan.GlmLayerFeedForwardPlan = match
      (try! spec.feed_forward_kind(layer)) {
      DenseSwiGlu => DenseSwiGlu
      RoutedMixtureOfExperts => RoutedMoe
    }
    layers.push(
      try! @glm_hybrid_execution_plan.GlmHybridLayerPlan::new(
        layer_index=layer,
        attention~,
        feed_forward~,
      ),
    )
  }
  layers
}

///|
fn flash_layers(
  spec : @glm53_spec.Glm53FlashSpec,
) -> Array[@glm_hybrid_execution_plan.GlmHybridLayerPlan] {
  let layers = []
  for layer in 0..<spec.layer_count() {
    let attention : @glm_hybrid_execution_plan.GlmLayerAttentionPlan = match
      (try! spec.attention_kind(layer)) {
      DeepSeekSparseAttention => DynamicSparse(ComputeTopK)
      KimiDeltaAttention => KimiDelta
    }
    let feed_forward : @glm_hybrid_execution_plan.GlmLayerFeedForwardPlan = match
      (try! spec.feed_forward_kind(layer)) {
      DenseSwiGlu => DenseSwiGlu
      RoutedMixtureOfExperts => RoutedMoe
    }
    layers.push(
      try! @glm_hybrid_execution_plan.GlmHybridLayerPlan::new(
        layer_index=layer,
        attention~,
        feed_forward~,
      ),
    )
  }
  layers
}

///|
fn flash_vision(
  spec : @glm53_spec.Glm53FlashSpec,
) -> @glm_hybrid_execution_plan.GlmVisionConditioningPlan {
  try! @glm_hybrid_execution_plan.GlmVisionConditioningPlan::new(
    layer_count=spec.vision_layer_count(),
    hidden_size=spec.vision_hidden_size(),
    intermediate_size=spec.vision_intermediate_size(),
    attention_head_count=spec.vision_head_count(),
    spatial_patch_size=spec.vision_patch_size(),
    temporal_patch_size=spec.vision_temporal_patch_size(),
    spatial_merge_size=spec.vision_spatial_merge_size(),
    decoder_projection_size=spec.vision_projection_size(),
    projection_intermediate_size=spec.vision_projection_intermediate_size(),
    vocabulary_size=spec.vocabulary_size(),
    image_token_id=spec.image_token_id(),
    video_token_id=spec.video_token_id(),
    image_start_token_id=spec.image_start_token_id(),
    image_end_token_id=spec.image_end_token_id(),
    video_start_token_id=spec.video_start_token_id(),
    video_end_token_id=spec.video_end_token_id(),
  )
}

///|
/// Builds the exact official text-only GLM-5.3 hybrid decoder plan.
pub fn build_full_execution_plan(
  metadata : @glm53_spec.Glm53FullMetadata,
) -> @glm_hybrid_execution_plan.GlmHybridExecutionPlan {
  let spec = metadata.spec()
  let auxiliary = try! @advanced_decoder_execution_plan.AuxiliaryPredictionPlan::multi_token_prediction(
    spec.mtp_layer_count(),
  )
  let layers = full_layers(spec)
  try! @glm_hybrid_execution_plan.GlmHybridExecutionPlan::new(
    content_digest=metadata.content_digest(),
    schema_version=1,
    profile_identity="zai-org/GLM-5.3:glm_moe_dsa",
    architecture=spec.architecture(),
    model_type=spec.model_type(),
    vocabulary_size=spec.vocabulary_size(),
    hidden_size=spec.hidden_size(),
    dense_intermediate_size=spec.dense_intermediate_size(),
    decoder_layer_count=spec.layer_count(),
    context_length=spec.context_length(),
    sparse_attention=full_sparse_attention(spec),
    kda=None,
    hyper_connection=None,
    moe=full_moe(spec),
    moe_router_float32=spec.uses_float32_moe_router(),
    moe_correction_bias=spec.uses_moe_correction_bias(),
    layers~,
    auxiliary_prediction=auxiliary,
    vision_conditioning=None,
  )
}

///|
/// Builds the exact official multimodal GLM-5.3-Flash hybrid decoder plan.
pub fn build_flash_execution_plan(
  metadata : @glm53_spec.Glm53FlashMetadata,
) -> @glm_hybrid_execution_plan.GlmHybridExecutionPlan {
  let spec = metadata.spec()
  let auxiliary = try! @advanced_decoder_execution_plan.AuxiliaryPredictionPlan::multi_token_prediction(
    spec.mtp_layer_count(),
  )
  let kda = try! @glm_hybrid_execution_plan.GlmKdaExecutionPlan::new(
    head_count=spec.linear_head_count(),
    head_dimension=spec.linear_head_dimension(),
    short_convolution_kernel_size=spec.linear_conv_kernel_size(),
    gate_lower_bound=spec.linear_gate_lower_bound(),
  )
  let hyper = try! @advanced_decoder_execution_plan.HyperConnectionPlan::new(
    width=spec.hyper_connection_streams(),
    control_epsilon=spec.hyper_connection_epsilon(),
    normalization_epsilon=spec.rms_norm_epsilon(),
    sinkhorn_iterations=spec.hyper_connection_sinkhorn_iterations(),
  )
  let layers = flash_layers(spec)
  try! @glm_hybrid_execution_plan.GlmHybridExecutionPlan::new(
    content_digest=metadata.content_digest(),
    schema_version=1,
    profile_identity="zai-org/GLM-5.3-Flash-BF16:glm5_next",
    architecture=spec.architecture(),
    model_type=spec.model_type(),
    vocabulary_size=spec.vocabulary_size(),
    hidden_size=spec.hidden_size(),
    dense_intermediate_size=spec.dense_intermediate_size(),
    decoder_layer_count=spec.layer_count(),
    context_length=spec.context_length(),
    sparse_attention=flash_sparse_attention(spec),
    kda=Some(kda),
    hyper_connection=Some(hyper),
    moe=flash_moe(spec),
    moe_router_float32=spec.uses_float32_moe_router(),
    moe_correction_bias=spec.uses_moe_correction_bias(),
    layers~,
    auxiliary_prediction=auxiliary,
    vision_conditioning=Some(flash_vision(spec)),
  )
}
