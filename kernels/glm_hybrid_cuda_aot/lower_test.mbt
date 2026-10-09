///|
fn glm_aot_requirements(
  pooled? : Bool = false,
  official_index? : Bool = false,
  official_sparse? : Bool = false,
  pool_tail? : Bool = true,
  rms_epsilon? : Double = 0.00001,
  rope_theta? : Double = 10000.0,
  kda_head_count? : Int = 0,
  routed_expert_count? : Int = 16,
  experts_per_token? : Int = 2,
  moe_intermediate_size? : Int = 32,
  moe_router_float32? : Bool = true,
  moe_activation? : @advanced_decoder_execution_plan.ExpertActivation = SwiGlu,
  hyper_connection? : Bool = false,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements {
  let sparse = try! @glm_hybrid_execution_plan.GlmDynamicSparseAttentionPlan::new(
    query_head_count=if official_sparse {
      64
    } else if official_index {
      32
    } else {
      8
    },
    kv_head_count=if official_sparse { 64 } else { 8 },
    query_lora_rank=if official_sparse {
      if pooled {
        1536
      } else {
        2048
      }
    } else {
      32
    },
    kv_lora_rank=if official_sparse { 512 } else { 16 },
    qk_nope_head_dimension=if official_sparse {
      if pooled {
        256
      } else {
        192
      }
    } else {
      16
    },
    qk_rope_head_dimension=if official_sparse {
      if pooled {
        0
      } else {
        64
      }
    } else {
      8
    },
    value_head_dimension=if official_sparse { 256 } else { 24 },
    rms_norm_epsilon=rms_epsilon,
    rope_theta=if official_sparse && pooled { None } else { Some(rope_theta) },
    rope_interleave=!(official_sparse && pooled),
    index_head_count=if official_index { 32 } else { 4 },
    index_head_dimension=if pooled || official_index { 128 } else { 8 },
    index_topk=if pooled || official_index { 2048 } else { 64 },
    indexer_rope_interleave=true,
    index_share_for_mtp_iteration=true,
    index_pooling=if pooled {
      Some(
        try! @glm_hybrid_execution_plan.GlmSparseIndexPoolingPlan::new(
          kernel_size=4,
          compress=true,
          always_select_tail=pool_tail,
        ),
      )
    } else {
      None
    },
  )
  let moe = try! @advanced_decoder_execution_plan.MoeExecutionPlan::new(
    routed_expert_count~,
    shared_expert_count=1,
    experts_per_token~,
    intermediate_size=moe_intermediate_size,
    routed_scaling_factor=2.5,
    normalize_selected_probabilities=true,
    score_function=Sigmoid,
    selection_method=NoAuxiliaryLossTokenChoice,
    activation=moe_activation,
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
  if kda_head_count > 0 {
    layers.push(
      try! @glm_hybrid_execution_plan.GlmHybridLayerPlan::new(
        layer_index=2,
        attention=KimiDelta,
        feed_forward=RoutedMoe,
      ),
    )
  }
  let kda = if kda_head_count > 0 {
    Some(
      try! @glm_hybrid_execution_plan.GlmKdaExecutionPlan::new(
        head_count=kda_head_count,
        head_dimension=128,
        short_convolution_kernel_size=4,
        gate_lower_bound=-5.0,
      ),
    )
  } else {
    None
  }
  let plan = try! @glm_hybrid_execution_plan.GlmHybridExecutionPlan::new(
    content_digest=try! @spec.ContentDigest::from_sha256(String::make(64, 'c')),
    schema_version=1,
    profile_identity="fixture/glm-hybrid-cuda-aot:v1",
    architecture="FixtureForCausalLM",
    model_type="fixture_glm",
    vocabulary_size=128,
    hidden_size=if official_sparse {
      if pooled {
        4096
      } else {
        6144
      }
    } else {
      64
    },
    dense_intermediate_size=if official_sparse { 12288 } else { 128 },
    decoder_layer_count=layers.length(),
    context_length=if pooled || official_index { 4096 } else { 1024 },
    sparse_attention=sparse,
    kda~,
    hyper_connection=if hyper_connection {
      Some(
        try! @advanced_decoder_execution_plan.HyperConnectionPlan::new(
          width=4,
          control_epsilon=0.000001,
          normalization_epsilon=0.00001,
          sinkhorn_iterations=20,
        ),
      )
    } else {
      None
    },
    moe~,
    moe_router_float32~,
    moe_correction_bias=true,
    layers~,
    auxiliary_prediction=try! @advanced_decoder_execution_plan.AuxiliaryPredictionPlan::multi_token_prediction(
      1,
    ),
    vision_conditioning=None,
  )
  try! @glm_hybrid_kernel_requirements.derive_requirements(
    plan,
    limits=try! @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementLimits::new(
      maximum_layers=4,
      maximum_vision_layers=4,
      maximum_requirements=64,
    ),
  )
}

///|
fn glm_aot_target(bf16? : Bool = true) -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=0,
    supports_bf16=bf16,
    supports_cublas_lt=true,
  )
}

///|
fn glm_aot_compiler(
  fused_multiply_add? : Bool = false,
  reassociate? : Bool = false,
) -> @luna_specialization.LunaCompilerPolicy {
  let identity = try! @luna_specialization.LunaCompilerIdentity::new(
    toolchain_digest_sha256=String::make(64, 'd'),
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
test "GLM language-model head binds exact dense projection geometry and ABI" {
  let requirements = glm_aot_requirements()
  let ordinal = requirements.requirements().length() - 1
  let candidate = lower_language_model_head_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(candidate.family(), LanguageModelHeadBf16)
  assert_eq(candidate.requirement_ordinal(), ordinal)
  assert_eq(candidate.operation(), LanguageModelHead(64, 128))
  assert_eq(candidate.input_width(), 64)
  assert_eq(candidate.output_width(), 128)
  assert_eq(candidate.maximum_tokens(), 32)
  assert_eq(candidate.dimensions().grid_x(), 16)
  assert_eq(candidate.dimensions().block_x(), 256)
  assert_eq(candidate.operands().length(), 4)
  assert_eq(candidate.operands()[0].role(), StepCountsInput)
  assert_eq(candidate.operands()[0].byte_count(), 20L)
  assert_eq(candidate.operands()[0].alignment(), 4L)
  assert_eq(candidate.operands()[1].byte_count(), 4096L)
  assert_eq(candidate.operands()[2].byte_count(), 16384L)
  assert_eq(candidate.operands()[3].role(), LogitsOutput)
  assert_eq(candidate.operands()[3].byte_count(), 8192L)
  assert_eq(
    candidate.numerical_contract(),
    OrderedFloat32AccumulationBf16RoundV1,
  )
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("const int row_count = counts[3];"))
  assert_true(source.contains("__fmul_rn"))
  assert_true(source.contains("__fadd_rn"))
  assert_false(source.contains("bias"))
  assert_true(
    @utf8.decode(candidate.recipe_bytes()).contains(
      "weights=row-major-output-input-bf16-no-bias",
    ),
  )
  assert_false(candidate.manifest_bindable())
  let repeated = lower_language_model_head_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(candidate.source_digest(), repeated.source_digest())
  assert_eq(candidate.recipe_digest(), repeated.recipe_digest())
}

///|
fn glm_aot_limits() -> GlmHybridCudaAotLimits {
  try! GlmHybridCudaAotLimits::new(
    maximum_tokens=4096,
    maximum_source_bytes=65536,
    maximum_recipe_bytes=65536,
  )
}

///|
test "GLM embedding decoder norm and final norm share deterministic source" {
  let requirements = glm_aot_requirements()
  let embedding = lower_foundation_cuda_aot_candidate(
    requirements,
    requirement_ordinal=0,
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(embedding.family(), TokenEmbeddingBf16)
  assert_eq(embedding.operation(), TokenEmbedding)
  assert_eq(embedding.operands()[0].role(), StepCountsInput)
  assert_eq(embedding.operands()[1].byte_count(), 128L)
  assert_false(embedding.manifest_bindable())
  let decoder_norm = lower_foundation_cuda_aot_candidate(
    requirements,
    requirement_ordinal=1,
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(decoder_norm.family(), DecoderRmsNormBf16)
  assert_eq(decoder_norm.operation(), DecoderRmsNorm(64, 0.00001))
  let final_ordinal = requirements.requirements().length() - 2
  let final_norm = lower_foundation_cuda_aot_candidate(
    requirements,
    requirement_ordinal=final_ordinal,
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(final_norm.family(), FinalRmsNormBf16)
  assert_eq(final_norm.operation(), FinalRmsNorm(64, 0.00001))
  assert_true(decoder_norm.source_bytes() != final_norm.source_bytes())
  assert_eq(embedding.source_digest().as_hex().length(), 64)
  assert_eq(final_norm.recipe_digest().as_hex().length(), 64)
  inspect(
    embedding.source_digest().as_hex(),
    content=(
      #|7b76e1603097b058e821e7d6ddabc6996735ae464966b283f110b7f0123f3921
    ),
  )
  inspect(
    embedding.recipe_digest().as_hex(),
    content=(
      #|c37e590ecb7b96c0a5b000d075a6fb89815c6c72516aba94ea453394440bd38d
    ),
  )
  inspect(
    decoder_norm.source_digest().as_hex(),
    content=(
      #|4283cfadf580abce568dd52c1a6dd2041e79dfc159b7f599573b05d47368b123
    ),
  )
  inspect(
    decoder_norm.recipe_digest().as_hex(),
    content=(
      #|7a75f2c7969db61421b866a01472999482487c4a66a8eef53db809bb8cb73d24
    ),
  )
  inspect(
    final_norm.source_digest().as_hex(),
    content=(
      #|6d1c54c183ae43ca0abef098f57a06f46c0a10176aec5c05331806e65fd997ec
    ),
  )
  inspect(
    final_norm.recipe_digest().as_hex(),
    content=(
      #|5ecba04443d5f9e048086104118c6f6b91e527fdffcee6528bc23e74d083c410
    ),
  )
  let repeated = lower_foundation_cuda_aot_candidate(
    requirements,
    requirement_ordinal=final_ordinal,
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(final_norm.source_digest(), repeated.source_digest())
  assert_eq(final_norm.recipe_digest(), repeated.recipe_digest())
}

///|
test "GLM lowerer rejects unsupported operations and unsafe policies" {
  let requirements = glm_aot_requirements()
  let lm_head_ordinal = requirements.requirements().length() - 1
  try
    lower_foundation_cuda_aot_candidate(
      requirements,
      requirement_ordinal=2,
      maximum_tokens=1,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    UnsupportedRequirement(2) => ()
    _ => fail("unexpected unsupported-operation rejection")
  } noraise {
    _ => fail("unsupported GLM operation was lowered")
  }
  try
    lower_foundation_cuda_aot_candidate(
      requirements,
      requirement_ordinal=0,
      maximum_tokens=1,
      target=glm_aot_target(bf16=false),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("unexpected target rejection")
  } noraise {
    _ => fail("non-BF16 GLM target was lowered")
  }
  try
    lower_foundation_cuda_aot_candidate(
      requirements,
      requirement_ordinal=0,
      maximum_tokens=1,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(reassociate=true),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("unexpected compiler rejection")
  } noraise {
    _ => fail("reassociating GLM compiler policy was accepted")
  }
  try
    lower_language_model_head_cuda_aot_candidate(
      requirements,
      requirement_ordinal=lm_head_ordinal,
      maximum_tokens=1,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(fused_multiply_add=true),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("unexpected LM-head compiler rejection")
  } noraise {
    _ => fail("FMA-enabled LM-head compiler policy was accepted")
  }
}
