///|
fn aot_test_requirements(
  score_function? : @advanced_decoder_execution_plan.ExpertScoreFunction = Sigmoid,
  content_fill? : Char = 'a',
  hash_layer_count? : Int = 0,
  routed_expert_count? : Int = 8,
  experts_per_token? : Int = 2,
  attention_head_count? : Int = 2,
  attention_head_dimension? : Int = 8,
  rope_head_dimension? : Int = 2,
  maximum_position_count? : Int = 1024,
  deterministic_compression_ratio? : Int = 0,
  hidden_size? : Int = 16,
  query_low_rank? : Int = 4,
  output_low_rank? : Int = 4,
  output_group_count? : Int = 1,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements {
  let shape = try! @advanced_decoder_execution_plan.AdvancedDecoderShape::new(
    vocabulary_size=128,
    hidden_size~,
    decoder_layer_count=1,
    attention_head_count~,
    kv_head_count=1,
    attention_head_dimension~,
    rope_head_dimension~,
    query_low_rank~,
    output_low_rank~,
    output_group_count~,
    maximum_position_count~,
  )
  let low_rank = try! @advanced_decoder_execution_plan.LowRankAttentionPlan::new(
    query_rank=query_low_rank,
    output_rank=output_low_rank,
    output_group_count~,
    query_key_norm_epsilon=0.000001,
  )
  let position = try! @advanced_decoder_execution_plan.PositionEncodingPlan::new(
    rope_theta=10000.0,
    compressed_rope_theta=20000.0,
    original_position_count=128,
    scaling_factor=8.0,
    beta_fast=32.0,
    beta_slow=1.0,
  )
  let hyper = try! @advanced_decoder_execution_plan.HyperConnectionPlan::new(
    width=2,
    control_epsilon=0.000001,
    normalization_epsilon=0.000001,
    sinkhorn_iterations=4,
  )
  let moe = try! @advanced_decoder_execution_plan.MoeExecutionPlan::new(
    routed_expert_count~,
    shared_expert_count=1,
    experts_per_token~,
    intermediate_size=32,
    routed_scaling_factor=1.5,
    normalize_selected_probabilities=true,
    score_function~,
    selection_method=if hash_layer_count == 0 {
      NoAuxiliaryLossTokenChoice
    } else {
      TokenHashThenNoAuxiliaryLossTokenChoice(hash_layer_count~)
    },
    activation=SwiGlu,
  )
  let plan = try! @advanced_decoder_execution_plan.AdvancedDecoderExecutionPlan::new(
    schema_version=1,
    profile_identity="test/advanced-decoder-cuda-aot/v1",
    content_digest=try! @spec.ContentDigest::from_sha256(
      String::make(64, content_fill),
    ),
    shape~,
    low_rank_attention=low_rank,
    position_encoding=position,
    hyper_connection=hyper,
    moe~,
    layers=[
      try! @advanced_decoder_execution_plan.AdvancedDecoderLayerPlan::new(
        layer_index=0,
        attention=if deterministic_compression_ratio == 0 {
          try! @advanced_decoder_execution_plan.AdvancedAttentionPlan::sliding_window_causal(
            64,
          )
        } else {
          try! @advanced_decoder_execution_plan.AdvancedAttentionPlan::deterministic_compressed(
            compression_ratio=deterministic_compression_ratio,
            window_size=64,
          )
        },
      ),
    ],
    auxiliary_prediction=try! @advanced_decoder_execution_plan.AuxiliaryPredictionPlan::multi_token_prediction(
      1,
    ),
  )
  try! @advanced_decoder_kernel_requirements.derive_requirements(
    plan,
    limits=try! @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementLimits::new(
      maximum_layers=4,
      maximum_auxiliary_stages=4,
      maximum_requirements=64,
    ),
  )
}

///|
fn aot_test_target(bf16? : Bool = true) -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=0,
    supports_bf16=bf16,
    supports_cublas_lt=true,
  )
}

///|
fn aot_test_compiler(
  reassociate? : Bool = false,
  fused_multiply_add? : Bool = false,
) -> @luna_specialization.LunaCompilerPolicy {
  let identity = try! @luna_specialization.LunaCompilerIdentity::new(
    toolchain_digest_sha256=String::make(64, 'b'),
    major=12,
    minor=9,
    patch=1,
  )
  try! @luna_specialization.LunaCompilerPolicy::new(
    identity~,
    optimization_level=3,
    fused_multiply_add~,
    reassociate~,
    max_registers=128,
  )
}

///|
fn aot_test_limits() -> AdvancedDecoderCudaAotLimits {
  try! AdvancedDecoderCudaAotLimits::new(
    maximum_tokens=4096,
    maximum_source_bytes=65536,
    maximum_recipe_bytes=65536,
  )
}

///|
test "embedding and final norm lower deterministically with exact ABI" {
  let requirements = aot_test_requirements()
  let embedding = lower_foundation_cuda_aot_candidate(
    requirements,
    requirement_ordinal=0,
    maximum_tokens=32,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(embedding.family(), TokenEmbeddingBf16)
  assert_eq(embedding.operation(), TokenEmbedding)
  assert_eq(embedding.capability(), EmbeddingLookupAot)
  assert_eq(embedding.operands().length(), 4)
  assert_eq(embedding.operands()[0].role(), StepCountsInput)
  assert_eq(embedding.operands()[0].byte_count(), 20L)
  assert_eq(embedding.operands()[1].role(), TokenIdsInput)
  assert_eq(embedding.operands()[1].byte_count(), 128L)
  assert_false(embedding.manifest_bindable())
  assert_eq(embedding.source_digest().as_hex().length(), 64)
  inspect(
    embedding.source_digest().as_hex(),
    content="07babff8f3dcd2a9aa99a628812d7b3ae05fbb96364fca97f95717702f394aa7",
  )
  inspect(
    embedding.recipe_digest().as_hex(),
    content=(
      #|3a5e30c5e658fcf1865ee10c659f6698705bb1ba004e0aa578f856ab83cc0b7a
    ),
  )
  let repeated = lower_foundation_cuda_aot_candidate(
    requirements,
    requirement_ordinal=0,
    maximum_tokens=32,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(embedding.source_digest(), repeated.source_digest())
  assert_eq(embedding.recipe_digest(), repeated.recipe_digest())
  let final_ordinal = requirements.requirements().length() - 2
  let norm = lower_foundation_cuda_aot_candidate(
    requirements,
    requirement_ordinal=final_ordinal,
    maximum_tokens=32,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(norm.family(), FinalRmsNormBf16)
  assert_eq(norm.operation(), FinalRmsNorm(0.000001))
  assert_eq(norm.dimensions().grid_x(), 32)
  assert_eq(norm.dimensions().block_x(), 256)
  assert_eq(norm.dimensions().shared_memory_bytes(), 1024)
  inspect(
    norm.source_digest().as_hex(),
    content="e54be9d65d853c29c848508f5e8b95439def7967fec467556665c65a6bb90506",
  )
  inspect(
    norm.recipe_digest().as_hex(),
    content=(
      #|7e3f1acb1cdd88d11b4e0e5d6eb931a3a4a590d9a34c8cd0c34f0dedd6340e21
    ),
  )
}

///|
test "unsupported operations targets compilers and token bounds fail closed" {
  let requirements = aot_test_requirements()
  for ordinal in [1, requirements.requirements().length() - 1] {
    try
      lower_foundation_cuda_aot_candidate(
        requirements,
        requirement_ordinal=ordinal,
        maximum_tokens=1,
        target=aot_test_target(),
        compiler=aot_test_compiler(),
        limits=aot_test_limits(),
      )
    catch {
      UnsupportedRequirement(actual) => assert_eq(actual, ordinal)
      _ => fail("unexpected unsupported-operation rejection")
    } noraise {
      _ => fail("unsupported advanced operation was lowered")
    }
  }
  try
    lower_foundation_cuda_aot_candidate(
      requirements,
      requirement_ordinal=0,
      maximum_tokens=4097,
      target=aot_test_target(),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("unexpected token-bound rejection")
  } noraise {
    _ => fail("oversized token profile was lowered")
  }
  try
    lower_foundation_cuda_aot_candidate(
      requirements,
      requirement_ordinal=0,
      maximum_tokens=1,
      target=aot_test_target(bf16=false),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("unexpected target rejection")
  } noraise {
    _ => fail("non-BF16 target was lowered")
  }
  try
    lower_foundation_cuda_aot_candidate(
      requirements,
      requirement_ordinal=0,
      maximum_tokens=1,
      target=aot_test_target(),
      compiler=aot_test_compiler(reassociate=true),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("unexpected compiler rejection")
  } noraise {
    _ => fail("reassociating compiler policy was lowered")
  }
}
