///|
fn fixture_content() -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
}

///|
fn fixture_sparse() -> @glm_hybrid_execution_plan.GlmDynamicSparseAttentionPlan {
  try! @glm_hybrid_execution_plan.GlmDynamicSparseAttentionPlan::new(
    query_head_count=8,
    kv_head_count=8,
    query_lora_rank=32,
    kv_lora_rank=16,
    qk_nope_head_dimension=16,
    qk_rope_head_dimension=8,
    value_head_dimension=24,
    rms_norm_epsilon=0.00001,
    rope_theta=Some(10000.0),
    rope_interleave=true,
    index_head_count=4,
    index_head_dimension=8,
    index_topk=64,
    indexer_rope_interleave=true,
    index_share_for_mtp_iteration=true,
    index_pooling=None,
  )
}

///|
fn fixture_moe() -> @advanced_decoder_execution_plan.MoeExecutionPlan {
  try! @advanced_decoder_execution_plan.MoeExecutionPlan::new(
    routed_expert_count=16,
    shared_expert_count=1,
    experts_per_token=2,
    intermediate_size=32,
    routed_scaling_factor=2.5,
    normalize_selected_probabilities=true,
    score_function=Sigmoid,
    selection_method=NoAuxiliaryLossTokenChoice,
    activation=SwiGlu,
  )
}

///|
fn fixture_layer(
  index : Int,
  attention : @glm_hybrid_execution_plan.GlmLayerAttentionPlan,
  feed_forward : @glm_hybrid_execution_plan.GlmLayerFeedForwardPlan,
) -> @glm_hybrid_execution_plan.GlmHybridLayerPlan {
  try! @glm_hybrid_execution_plan.GlmHybridLayerPlan::new(
    layer_index=index,
    attention~,
    feed_forward~,
  )
}

///|
fn fixture_plan(
  layers : ArrayView[@glm_hybrid_execution_plan.GlmHybridLayerPlan],
  decoder_layer_count : Int,
  kda : @glm_hybrid_execution_plan.GlmKdaExecutionPlan?,
) -> @glm_hybrid_execution_plan.GlmHybridExecutionPlan raise @glm_hybrid_execution_plan.GlmHybridExecutionPlanError {
  @glm_hybrid_execution_plan.GlmHybridExecutionPlan::new(
    content_digest=fixture_content(),
    schema_version=1,
    profile_identity="fixture/glm-hybrid:v1",
    architecture="FixtureForCausalLM",
    model_type="fixture_glm",
    vocabulary_size=128,
    hidden_size=64,
    dense_intermediate_size=128,
    decoder_layer_count~,
    context_length=1024,
    sparse_attention=fixture_sparse(),
    kda~,
    hyper_connection=None,
    moe=fixture_moe(),
    moe_router_float32=true,
    moe_correction_bias=true,
    layers~,
    auxiliary_prediction=try! @advanced_decoder_execution_plan.AuxiliaryPredictionPlan::multi_token_prediction(
      1,
    ),
    vision_conditioning=None,
  )
}

///|
test "GLM hybrid plan identity is deterministic and schedule-sensitive" {
  let layers = [
    fixture_layer(0, DynamicSparse(ComputeTopK), DenseSwiGlu),
    fixture_layer(1, DynamicSparse(ReusePreviousTopK), RoutedMoe),
  ]
  let first = fixture_plan(layers, 2, None)
  let second = fixture_plan(layers, 2, None)
  assert_eq(first.identity(), second.identity())
  inspect(
    first.identity().plan().as_hex(),
    content="da85c2df01f97a3bf9bf0c6a4579d16b9d70a86b2167f9464393ec34763c4213",
  )
  let changed = fixture_plan(
    [
      fixture_layer(0, DynamicSparse(ComputeTopK), DenseSwiGlu),
      fixture_layer(1, DynamicSparse(ComputeTopK), RoutedMoe),
    ],
    2,
    None,
  )
  assert_true(first.identity().plan() != changed.identity().plan())
}

///|
test "GLM hybrid plan rejects shared sparse index before computation" {
  try
    fixture_plan(
      [
        fixture_layer(0, DynamicSparse(ReusePreviousTopK), DenseSwiGlu),
        fixture_layer(1, DynamicSparse(ComputeTopK), RoutedMoe),
      ],
      2,
      None,
    )
  catch {
    InvalidSparseIndexSchedule => ()
    _ => fail("unexpected sparse-index validation error")
  } noraise {
    _ => fail("shared sparse index without producer was admitted")
  }
}

///|
test "GLM hybrid plan rejects missing KDA state geometry" {
  try
    fixture_plan(
      [
        fixture_layer(0, DynamicSparse(ComputeTopK), DenseSwiGlu),
        fixture_layer(1, KimiDelta, RoutedMoe),
      ],
      2,
      None,
    )
  catch {
    MissingKdaContract => ()
    _ => fail("unexpected KDA validation error")
  } noraise {
    _ => fail("KDA layer without state geometry was admitted")
  }
}

///|
test "GLM hybrid plan rejects layer count and order drift" {
  let layers = [
    fixture_layer(0, DynamicSparse(ComputeTopK), DenseSwiGlu),
    fixture_layer(2, DynamicSparse(ReusePreviousTopK), RoutedMoe),
  ]
  try fixture_plan(layers, 3, None) catch {
    LayerCountMismatch => ()
    _ => fail("unexpected layer-count validation error")
  } noraise {
    _ => fail("truncated layer schedule was admitted")
  }
  try fixture_plan(layers, 2, None) catch {
    InvalidLayerOrder => ()
    _ => fail("unexpected layer-order validation error")
  } noraise {
    _ => fail("reordered layer schedule was admitted")
  }
}

///|
test "vision conditioning rejects aliased control tokens" {
  try
    @glm_hybrid_execution_plan.GlmVisionConditioningPlan::new(
      layer_count=2,
      hidden_size=32,
      intermediate_size=64,
      attention_head_count=4,
      spatial_patch_size=14,
      temporal_patch_size=2,
      spatial_merge_size=2,
      decoder_projection_size=64,
      projection_intermediate_size=128,
      vocabulary_size=128,
      image_token_id=100,
      video_token_id=100,
      image_start_token_id=101,
      image_end_token_id=102,
      video_start_token_id=103,
      video_end_token_id=104,
    )
  catch {
    InvalidVisionConditioning => ()
    _ => fail("unexpected vision conditioning error")
  } noraise {
    _ => fail("aliased multimodal control tokens were admitted")
  }
}

///|
test "DSA and KDA constructors reject contradictory state geometry" {
  try
    @glm_hybrid_execution_plan.GlmDynamicSparseAttentionPlan::new(
      query_head_count=8,
      kv_head_count=8,
      query_lora_rank=32,
      kv_lora_rank=16,
      qk_nope_head_dimension=16,
      qk_rope_head_dimension=0,
      value_head_dimension=24,
      rms_norm_epsilon=0.00001,
      rope_theta=None,
      rope_interleave=true,
      index_head_count=4,
      index_head_dimension=8,
      index_topk=64,
      indexer_rope_interleave=true,
      index_share_for_mtp_iteration=true,
      index_pooling=None,
    )
  catch {
    InvalidSparseAttentionGeometry => ()
    _ => fail("unexpected DSA geometry error")
  } noraise {
    _ => fail("zero-RoPE DSA admitted interleaving")
  }
  try
    @glm_hybrid_execution_plan.GlmKdaExecutionPlan::new(
      head_count=8,
      head_dimension=16,
      short_convolution_kernel_size=0,
      gate_lower_bound=-5.0,
    )
  catch {
    InvalidKdaGeometry => ()
    _ => fail("unexpected KDA geometry error")
  } noraise {
    _ => fail("KDA admitted an empty convolution state")
  }
}
