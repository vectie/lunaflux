///|
fn test_shape(
  layer_count : Int,
) -> @advanced_decoder_execution_plan.AdvancedDecoderShape raise @advanced_decoder_execution_plan.AdvancedDecoderPlanError {
  @advanced_decoder_execution_plan.AdvancedDecoderShape::new(
    vocabulary_size=128,
    hidden_size=64,
    decoder_layer_count=layer_count,
    attention_head_count=8,
    kv_head_count=1,
    attention_head_dimension=16,
    rope_head_dimension=8,
    query_low_rank=32,
    output_low_rank=16,
    output_group_count=4,
    maximum_position_count=4096,
  )
}

///|
fn test_plan(
  reverse_attention : Bool,
) -> @advanced_decoder_execution_plan.AdvancedDecoderExecutionPlan raise {
  let sliding = @advanced_decoder_execution_plan.AdvancedAttentionPlan::sliding_window_causal(
    128,
  )
  let compressed = @advanced_decoder_execution_plan.AdvancedAttentionPlan::indexed_compressed(
    compression_ratio=4,
    index_head_count=8,
    index_head_dimension=8,
    index_top_k=64,
  )
  let first = if reverse_attention { compressed } else { sliding }
  let second = if reverse_attention { sliding } else { compressed }
  @advanced_decoder_execution_plan.AdvancedDecoderExecutionPlan::new(
    schema_version=1,
    profile_identity="fixture/advanced-decoder/v1",
    content_digest=@spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    shape=test_shape(2),
    low_rank_attention=@advanced_decoder_execution_plan.LowRankAttentionPlan::new(
      query_rank=32,
      output_rank=16,
      output_group_count=4,
      query_key_norm_epsilon=0.000001,
    ),
    position_encoding=@advanced_decoder_execution_plan.PositionEncodingPlan::new(
      rope_theta=10000.0,
      compressed_rope_theta=160000.0,
      original_position_count=1024,
      scaling_factor=4.0,
      beta_fast=32.0,
      beta_slow=1.0,
    ),
    hyper_connection=@advanced_decoder_execution_plan.HyperConnectionPlan::new(
      width=4,
      control_epsilon=0.000001,
      normalization_epsilon=0.000001,
      sinkhorn_iterations=20,
    ),
    moe=@advanced_decoder_execution_plan.MoeExecutionPlan::new(
      routed_expert_count=16,
      shared_expert_count=1,
      experts_per_token=2,
      intermediate_size=128,
      routed_scaling_factor=1.5,
      normalize_selected_probabilities=true,
      score_function=SqrtSoftplus,
      selection_method=NoAuxiliaryLossTokenChoice,
      activation=ClampedSwiGlu(10.0),
    ),
    layers=[
      @advanced_decoder_execution_plan.AdvancedDecoderLayerPlan::new(
        layer_index=0,
        attention=first,
      ),
      @advanced_decoder_execution_plan.AdvancedDecoderLayerPlan::new(
        layer_index=1,
        attention=second,
      ),
    ],
    auxiliary_prediction=@advanced_decoder_execution_plan.AuxiliaryPredictionPlan::multi_token_prediction(
      1,
    ),
  )
}

///|
test "advanced decoder plan identity is deterministic and semantic" {
  let first = test_plan(false)
  let second = test_plan(false)
  assert_eq(first, second)
  inspect(
    first.plan_digest().as_hex(),
    content=(
      #|1444eebe6ea5bffedc6327fda9f986741fbfa16ababe54ff437442e40751b5c2
    ),
  )
  assert_true(first.plan_digest() != test_plan(true).plan_digest())
}

///|
test "advanced decoder shape rejects incompatible output grouping" {
  try
    @advanced_decoder_execution_plan.AdvancedDecoderShape::new(
      vocabulary_size=128,
      hidden_size=64,
      decoder_layer_count=2,
      attention_head_count=8,
      kv_head_count=1,
      attention_head_dimension=16,
      rope_head_dimension=8,
      query_low_rank=32,
      output_low_rank=16,
      output_group_count=3,
      maximum_position_count=4096,
    )
  catch {
    InvalidAttentionGeometry => ()
    _ => fail("unexpected advanced decoder shape error")
  } noraise {
    _ => fail("incompatible output grouping was admitted")
  }
}

///|
test "advanced decoder plan rejects reordered and missing layers" {
  let attention = @advanced_decoder_execution_plan.AdvancedAttentionPlan::sliding_window_causal(
    128,
  )
  let layer = @advanced_decoder_execution_plan.AdvancedDecoderLayerPlan::new(
    layer_index=1,
    attention~,
  )
  let base = test_plan(false)
  try
    @advanced_decoder_execution_plan.AdvancedDecoderExecutionPlan::new(
      schema_version=base.schema_version(),
      profile_identity=base.profile_identity(),
      content_digest=base.content_digest(),
      shape=test_shape(1),
      low_rank_attention=base.low_rank_attention(),
      position_encoding=base.position_encoding(),
      hyper_connection=base.hyper_connection(),
      moe=base.moe(),
      layers=[layer],
      auxiliary_prediction=base.auxiliary_prediction(),
    )
  catch {
    InvalidLayerOrder => ()
    _ => fail("unexpected advanced decoder layer error")
  } noraise {
    _ => fail("reordered layer was admitted")
  }
}

///|
test "DSpark target layers are strictly ordered and bounded" {
  for targets in [[2, 2], [3], [-1]] {
    try
      @advanced_decoder_execution_plan.DsparkExecutionPlan::new(
        block_size=5,
        noise_token_id=127,
        target_layer_ids=targets,
        markov_rank=16,
        decoder_layer_count=3,
      )
    catch {
      InvalidAuxiliaryPrediction => ()
      _ => fail("unexpected DSpark validation error")
    } noraise {
      _ => fail("invalid DSpark target set was admitted")
    }
  }
}

///|
test "public auxiliary constructors cannot bypass plan validation" {
  let base = test_plan(false)
  try
    @advanced_decoder_execution_plan.AdvancedDecoderExecutionPlan::new(
      schema_version=base.schema_version(),
      profile_identity=base.profile_identity(),
      content_digest=base.content_digest(),
      shape=base.shape(),
      low_rank_attention=base.low_rank_attention(),
      position_encoding=base.position_encoding(),
      hyper_connection=base.hyper_connection(),
      moe=base.moe(),
      layers=base.layers(),
      auxiliary_prediction=MultiTokenPrediction(layer_count=0),
    )
  catch {
    InvalidAuxiliaryPrediction => ()
    _ => fail("unexpected forged auxiliary prediction error")
  } noraise {
    _ => fail("forged zero-layer auxiliary prediction was admitted")
  }
}

///|
test "hash-routed MoE layers cannot exceed the decoder" {
  let base = test_plan(false)
  let hostile_moe = @advanced_decoder_execution_plan.MoeExecutionPlan::new(
    routed_expert_count=16,
    shared_expert_count=1,
    experts_per_token=2,
    intermediate_size=128,
    routed_scaling_factor=1.5,
    normalize_selected_probabilities=true,
    score_function=SqrtSoftplus,
    selection_method=TokenHashThenNoAuxiliaryLossTokenChoice(hash_layer_count=3),
    activation=ClampedSwiGlu(10.0),
  )
  try
    @advanced_decoder_execution_plan.AdvancedDecoderExecutionPlan::new(
      schema_version=base.schema_version(),
      profile_identity=base.profile_identity(),
      content_digest=base.content_digest(),
      shape=base.shape(),
      low_rank_attention=base.low_rank_attention(),
      position_encoding=base.position_encoding(),
      hyper_connection=base.hyper_connection(),
      moe=hostile_moe,
      layers=base.layers(),
      auxiliary_prediction=base.auxiliary_prediction(),
    )
  catch {
    InvalidExpertRouting => ()
    _ => fail("unexpected hash-routed MoE error")
  } noraise {
    _ => fail("too many hash-routed layers were admitted")
  }
}
