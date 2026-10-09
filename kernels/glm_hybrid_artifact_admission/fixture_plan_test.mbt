///|
fn artifact_test_plan(
  fill? : Char = 'a',
) -> @glm_hybrid_execution_plan.GlmHybridExecutionPlan {
  let sparse = try! @glm_hybrid_execution_plan.GlmDynamicSparseAttentionPlan::new(
    query_head_count=2,
    kv_head_count=1,
    query_lora_rank=4,
    kv_lora_rank=2,
    qk_nope_head_dimension=4,
    qk_rope_head_dimension=2,
    value_head_dimension=4,
    rms_norm_epsilon=0.000001,
    rope_theta=Some(10000.0),
    rope_interleave=true,
    index_head_count=2,
    index_head_dimension=4,
    index_topk=8,
    indexer_rope_interleave=true,
    index_share_for_mtp_iteration=true,
    index_pooling=Some(
      try! @glm_hybrid_execution_plan.GlmSparseIndexPoolingPlan::new(
        kernel_size=2,
        compress=true,
        always_select_tail=true,
      ),
    ),
  )
  let moe = try! @advanced_decoder_execution_plan.MoeExecutionPlan::new(
    routed_expert_count=8,
    shared_expert_count=1,
    experts_per_token=2,
    intermediate_size=16,
    routed_scaling_factor=1.0,
    normalize_selected_probabilities=true,
    score_function=Sigmoid,
    selection_method=NoAuxiliaryLossTokenChoice,
    activation=SwiGlu,
  )
  let layers = [
    try! @glm_hybrid_execution_plan.GlmHybridLayerPlan::new(
      layer_index=0,
      attention=DynamicSparse(ComputeTopK),
      feed_forward=DenseSwiGlu,
    ),
    try! @glm_hybrid_execution_plan.GlmHybridLayerPlan::new(
      layer_index=1,
      attention=DynamicSparse(ReusePreviousTopK),
      feed_forward=RoutedMoe,
    ),
  ]
  try! @glm_hybrid_execution_plan.GlmHybridExecutionPlan::new(
    content_digest=try! @spec.ContentDigest::from_sha256(
      fill.to_string().repeat(64),
    ),
    schema_version=1,
    profile_identity="test/glm-hybrid-artifact",
    architecture="GlmHybridFixture",
    model_type="glm_hybrid_fixture",
    vocabulary_size=64,
    hidden_size=8,
    dense_intermediate_size=16,
    decoder_layer_count=2,
    context_length=32,
    sparse_attention=sparse,
    kda=None,
    hyper_connection=None,
    moe~,
    moe_router_float32=true,
    moe_correction_bias=true,
    layers~,
    auxiliary_prediction=try! @advanced_decoder_execution_plan.AuxiliaryPredictionPlan::multi_token_prediction(
      1,
    ),
    vision_conditioning=None,
  )
}
