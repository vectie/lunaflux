///|
fn kda_abi_requirements(
  routed_expert_count? : Int = 8,
  hyper_connection? : Bool = false,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements {
  let sparse = try! @glm_hybrid_execution_plan.GlmDynamicSparseAttentionPlan::new(
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
    index_topk=8,
    indexer_rope_interleave=true,
    index_share_for_mtp_iteration=true,
    index_pooling=None,
  )
  let kda = try! @glm_hybrid_execution_plan.GlmKdaExecutionPlan::new(
    head_count=64,
    head_dimension=128,
    short_convolution_kernel_size=4,
    gate_lower_bound=-5.0,
  )
  let moe = try! @advanced_decoder_execution_plan.MoeExecutionPlan::new(
    routed_expert_count~,
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
      attention=KimiDelta,
      feed_forward=DenseSwiGlu,
    ),
    try! @glm_hybrid_execution_plan.GlmHybridLayerPlan::new(
      layer_index=1,
      attention=KimiDelta,
      feed_forward=RoutedMoe,
    ),
  ]
  let plan = try! @glm_hybrid_execution_plan.GlmHybridExecutionPlan::new(
    content_digest=try! @spec.ContentDigest::from_sha256(String::make(64, 'e')),
    schema_version=1,
    profile_identity="fixture/kda-short-convolution-abi:v1",
    architecture="FixtureForCausalLM",
    model_type="fixture_glm_kda",
    vocabulary_size=128,
    hidden_size=4096,
    dense_intermediate_size=12288,
    decoder_layer_count=2,
    context_length=4096,
    sparse_attention=sparse,
    kda=Some(kda),
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
    moe_router_float32=true,
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
fn kda_abi_operand(
  role : GlmHybridLaunchOperandRole,
  bytes : Int64,
  alignment : Int64,
) -> GlmHybridLaunchOperand {
  try! GlmHybridLaunchOperand::new(role~, byte_count=bytes, alignment~)
}

///|
fn kda_abi_claim(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  operation? : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation = KdaShortConvolution(
    8192, 4,
  ),
  output_role? : GlmHybridLaunchOperandRole = KdaConvolvedOutput,
  final_state_bytes? : Int64 = 393216L,
  offsets_bytes? : Int64 = 36L,
  grid_x? : Int = 1024,
  symbol? : String = "lunaflux_glm_hybrid_kda_short_convolution_bf16_v1",
) -> GlmHybridArtifactClaimInput {
  let digest = try! @catalog.AotArtifactDigest::from_sha256(
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  )
  let family = @catalog.AotKernelFamily::new(
    module_digest=digest,
    family_id=try! @catalog.AotKernelFamilyId::from_stable_int(1),
  )
  let entry = @catalog.AotKernelEntryPoint::new(
    family~,
    entry_point_id=try! @catalog.AotKernelEntryPointId::from_stable_int(1),
  )
  let launch = try! @launch_contract.AotLaunchDimensions::new(
    grid_x~,
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  GlmHybridArtifactClaimInput::new(
    ordinal=0,
    source_digest=requirements.digest(),
    source_identity=requirements.source_identity(),
    target=try! @catalog.DeviceTarget::new(
      compute_major=9,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=true,
    ),
    operation~,
    capability=KdaShortConvolutionAot,
    module_digest=digest,
    module_bytes=b"abc",
    entry_point=entry,
    function_symbol=symbol,
    launch~,
    operands=[
      kda_abi_operand(StepCountsInput, 20L, 4L),
      kda_abi_operand(KdaSequenceOffsetsInput, offsets_bytes, 4L),
      kda_abi_operand(KdaProjectedInput, 524288L, 16L),
      kda_abi_operand(KdaShortConvolutionWeights, 65536L, 16L),
      kda_abi_operand(KdaConvolutionInitialStateInput, 393216L, 16L),
      kda_abi_operand(output_role, 524288L, 16L),
      kda_abi_operand(KdaConvolutionFinalStateOutput, final_state_bytes, 16L),
    ],
  )
}

///|
test "KDA short convolution admission authenticates exact ABI and hostile mismatches" {
  let requirements = kda_abi_requirements()
  assert_true(
    exact_supported_launch_abi(requirements, kda_abi_claim(requirements)),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kda_abi_claim(requirements, output_role=HiddenOutput),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kda_abi_claim(requirements, final_state_bytes=393214L),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kda_abi_claim(requirements, offsets_bytes=32L),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kda_abi_claim(requirements, grid_x=1023),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kda_abi_claim(requirements, symbol="wrong_symbol"),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kda_abi_claim(requirements, operation=KdaShortConvolution(8192, 3)),
    ),
  )
}
