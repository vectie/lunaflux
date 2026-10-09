///|
fn requirement_content(fill : Char) -> @spec.ContentDigest raise {
  @spec.ContentDigest::from_sha256(fill.to_string().repeat(64))
}

///|
fn requirement_plan(
  schema_version : Int,
  content : @spec.ContentDigest,
) -> @advanced_decoder_execution_plan.AdvancedDecoderExecutionPlan raise {
  let shape = @advanced_decoder_execution_plan.AdvancedDecoderShape::new(
    vocabulary_size=128,
    hidden_size=16,
    decoder_layer_count=2,
    attention_head_count=2,
    kv_head_count=1,
    attention_head_dimension=8,
    rope_head_dimension=2,
    query_low_rank=4,
    output_low_rank=4,
    output_group_count=1,
    maximum_position_count=1024,
  )
  let low_rank = @advanced_decoder_execution_plan.LowRankAttentionPlan::new(
    query_rank=4,
    output_rank=4,
    output_group_count=1,
    query_key_norm_epsilon=0.001,
  )
  let position = @advanced_decoder_execution_plan.PositionEncodingPlan::new(
    rope_theta=10000.0,
    compressed_rope_theta=20000.0,
    original_position_count=128,
    scaling_factor=8.0,
    beta_fast=32.0,
    beta_slow=1.0,
  )
  let hyper = @advanced_decoder_execution_plan.HyperConnectionPlan::new(
    width=2,
    control_epsilon=0.000001,
    normalization_epsilon=0.001,
    sinkhorn_iterations=4,
  )
  let moe = @advanced_decoder_execution_plan.MoeExecutionPlan::new(
    routed_expert_count=8,
    shared_expert_count=1,
    experts_per_token=2,
    intermediate_size=32,
    routed_scaling_factor=1.5,
    normalize_selected_probabilities=true,
    score_function=Sigmoid,
    selection_method=NoAuxiliaryLossTokenChoice,
    activation=SwiGlu,
  )
  let layers = [
    @advanced_decoder_execution_plan.AdvancedDecoderLayerPlan::new(
      layer_index=0,
      attention=@advanced_decoder_execution_plan.AdvancedAttentionPlan::sliding_window_causal(
        64,
      ),
    ),
    @advanced_decoder_execution_plan.AdvancedDecoderLayerPlan::new(
      layer_index=1,
      attention=@advanced_decoder_execution_plan.AdvancedAttentionPlan::indexed_compressed(
        compression_ratio=4,
        index_head_count=2,
        index_head_dimension=4,
        index_top_k=16,
      ),
    ),
  ]
  let auxiliary = @advanced_decoder_execution_plan.AuxiliaryPredictionPlan::multi_token_prediction(
    2,
  )
  @advanced_decoder_execution_plan.AdvancedDecoderExecutionPlan::new(
    schema_version~,
    profile_identity="test/advanced-kernel-requirements/v1",
    content_digest=content,
    shape~,
    low_rank_attention=low_rank,
    position_encoding=position,
    hyper_connection=hyper,
    moe~,
    layers~,
    auxiliary_prediction=auxiliary,
  )
}

///|
fn requirement_limits() -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementLimits raise {
  @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementLimits::new(
    maximum_layers=8,
    maximum_auxiliary_stages=8,
    maximum_requirements=64,
  )
}

///|
test "derivation is exact ordered unique and digest bound" {
  let plan = requirement_plan(1, requirement_content('a'))
  let first = @advanced_decoder_kernel_requirements.derive_requirements(
    plan,
    limits=requirement_limits(),
  )
  let second = @advanced_decoder_kernel_requirements.derive_requirements(
    plan,
    limits=requirement_limits(),
  )
  assert_eq(first, second)
  assert_eq(first.source_identity().content(), plan.content_digest())
  assert_eq(first.source_identity().plan(), plan.plan_digest())
  assert_eq(first.source_schema_version(), 1)
  assert_eq(first.geometry().vocabulary_size(), 128)
  assert_eq(first.geometry().hidden_size(), 16)
  assert_eq(first.geometry().decoder_layer_count(), 2)
  assert_eq(first.geometry().attention_head_count(), 2)
  assert_eq(first.geometry().kv_head_count(), 1)
  assert_eq(first.geometry().attention_head_dimension(), 8)
  assert_eq(first.geometry().rope_head_dimension(), 2)
  assert_eq(first.geometry().maximum_position_count(), 1024)
  assert_eq(first.requirements().length(), 27)
  assert_eq(first.requirements()[0].operation(), TokenEmbedding)
  assert_eq(first.requirements()[1].operation(), QueryLowRankAProjection(4))
  assert_eq(first.requirements()[3].operation(), QueryKeyRmsNorm(0.001, 4, 8))
  assert_eq(
    first.requirements()[4].operation(),
    QueryLowRankBProjection(4, 2, 8),
  )
  assert_eq(
    first.requirements()[6].operation(),
    HyperConnectionBlockControlV1(2, 4, 0.000001, 0.001),
  )
  assert_eq(first.requirements()[7].operation(), HyperConnectionPreReduceV1(2))
  assert_eq(
    first.requirements()[8].operation(),
    HyperConnectionPostCombineV1(2),
  )
  assert_eq(
    first.requirements()[9].operation(),
    HyperConnectionHeadReduceV1(2, 0.000001, 0.001),
  )
  assert_eq(
    first.requirements()[10].operation(),
    SlidingWindowCausalAttention(64),
  )
  assert_eq(
    first.requirements()[11].operation(),
    LearnedCompressedIndex(4, 2, 4, 16),
  )
  assert_eq(first.requirements()[22].operation(), AuxiliaryDecoderStage(0))
  assert_eq(first.requirements()[23].operation(), AuxiliaryDecoderStage(1))
  assert_eq(first.requirements()[25].operation(), FinalRmsNorm(0.001))
  assert_eq(first.requirements()[26].operation(), LanguageModelHead)
  assert_eq(first.capabilities().length(), 25)
  inspect(
    first.digest().as_hex(),
    content=(
      #|4468729a39a24f11700defdb1d46887eb3dd12c63b9c08ff4760b7d5ff2008b5
    ),
  )
}

///|
test "content and plan identity perturb the requirement digest" {
  let left = @advanced_decoder_kernel_requirements.derive_requirements(
    requirement_plan(1, requirement_content('a')),
    limits=requirement_limits(),
  )
  let right = @advanced_decoder_kernel_requirements.derive_requirements(
    requirement_plan(1, requirement_content('b')),
    limits=requirement_limits(),
  )
  assert_true(left.source_identity() != right.source_identity())
  assert_true(left.digest() != right.digest())
}

///|
test "current dense generic kernel vocabulary fails closed at first advanced operation" {
  let requirements = @advanced_decoder_kernel_requirements.derive_requirements(
    requirement_plan(1, requirement_content('c')),
    limits=requirement_limits(),
  )
  assert_eq(
    requirements.requirements()[0].current_generic_capability(ordinal=0),
    @plan.KernelCapabilityId::embedding_lookup(),
  )
  try requirements.require_current_generic_capabilities() catch {
    UnsupportedGenericKernelOperation(1, QueryLowRankAProjection(4)) => ()
    _ => fail("generic vocabulary returned the wrong rejection")
  } noraise {
    _ => fail("advanced requirements were weakened into dense capabilities")
  }
}

///|
test "hostile schema and derivation bounds fail before partial requirements escape" {
  try
    @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementLimits::new(
      maximum_layers=0,
      maximum_auxiliary_stages=1,
      maximum_requirements=1,
    )
  catch {
    InvalidLimit => ()
    _ => fail("invalid limit returned the wrong error")
  } noraise {
    _ => fail("zero layer limit was admitted")
  }
  let unknown_schema = requirement_plan(2, requirement_content('d'))
  let normal_limits = requirement_limits()
  try
    @advanced_decoder_kernel_requirements.derive_requirements(
      unknown_schema,
      limits=normal_limits,
    )
  catch {
    UnsupportedSchema => ()
    _ => fail("schema mutation returned the wrong error")
  } noraise {
    _ => fail("unknown requirements schema was admitted")
  }
  let low_layer_limit = @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementLimits::new(
    maximum_layers=1,
    maximum_auxiliary_stages=8,
    maximum_requirements=64,
  )
  let bounded_plan = requirement_plan(1, requirement_content('e'))
  try
    @advanced_decoder_kernel_requirements.derive_requirements(
      bounded_plan,
      limits=low_layer_limit,
    )
  catch {
    PlanExceedsLimits => ()
    _ => fail("layer bound returned the wrong error")
  } noraise {
    _ => fail("oversized layer set was admitted")
  }
  let low_requirement_limit = @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementLimits::new(
    maximum_layers=8,
    maximum_auxiliary_stages=8,
    maximum_requirements=2,
  )
  let requirement_bounded_plan = requirement_plan(1, requirement_content('f'))
  try
    @advanced_decoder_kernel_requirements.derive_requirements(
      requirement_bounded_plan,
      limits=low_requirement_limit,
    )
  catch {
    PlanExceedsLimits => ()
    _ => fail("requirement bound returned the wrong error")
  } noraise {
    _ => fail("oversized requirement set was admitted")
  }
}
