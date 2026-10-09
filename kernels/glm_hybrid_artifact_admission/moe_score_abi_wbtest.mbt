///|
fn moe_score_claim(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  output_role? : GlmHybridLaunchOperandRole = MoeUncorrectedScoresOutput,
  choice_role? : GlmHybridLaunchOperandRole = MoeChoiceScoresOutput,
  output_bytes? : Int64 = 36864L,
  symbol? : String = "lunaflux_glm_hybrid_moe_sigmoid_correction_f32_v1",
) -> GlmHybridArtifactClaimInput {
  let digest = try! @catalog.AotArtifactDigest::from_sha256(
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  )
  let family = @catalog.AotKernelFamily::new(
    module_digest=digest,
    family_id=try! @catalog.AotKernelFamilyId::from_stable_int(1),
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
    operation=MoeSigmoidAndCorrectionBias(288),
    capability=MoeSigmoidAndCorrectionBiasAot,
    module_digest=digest,
    module_bytes=b"abc",
    entry_point=@catalog.AotKernelEntryPoint::new(
      family~,
      entry_point_id=try! @catalog.AotKernelEntryPointId::from_stable_int(1),
    ),
    function_symbol=symbol,
    launch=try! @launch_contract.AotLaunchDimensions::new(
      grid_x=36,
      grid_y=1,
      grid_z=1,
      block_x=256,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    operands=[
      kda_abi_operand(StepCountsInput, 20L, 4L),
      kda_abi_operand(MoeRouterFloat32LogitsOutput, 36864L, 16L),
      kda_abi_operand(MoeCorrectionBiasInput, 1152L, 16L),
      kda_abi_operand(output_role, output_bytes, 16L),
      kda_abi_operand(choice_role, 36864L, 16L),
    ],
  )
}

///|
test "MoE score admission authenticates distinct uncorrected and choice outputs" {
  let requirements = kda_abi_requirements(routed_expert_count=288)
  assert_true(
    exact_supported_launch_abi(requirements, moe_score_claim(requirements)),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      moe_score_claim(
        requirements,
        output_role=MoeChoiceScoresOutput,
        choice_role=MoeChoiceScoresOutput,
      ),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      moe_score_claim(requirements, output_bytes=36860L),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      moe_score_claim(requirements, symbol="wrong_symbol"),
    ),
  )
}
