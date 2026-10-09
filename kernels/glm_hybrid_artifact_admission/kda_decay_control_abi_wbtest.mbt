///|
fn kda_decay_control_claim(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  bias_role? : GlmHybridLaunchOperandRole = KdaDecayBiasInput,
  output_bytes? : Int64 = 1048576L,
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
    operation=KdaDecayControl(64, 128, -5.0),
    capability=KdaDecayControlAot,
    module_digest=digest,
    module_bytes=b"abc",
    entry_point=@catalog.AotKernelEntryPoint::new(
      family~,
      entry_point_id=try! @catalog.AotKernelEntryPointId::from_stable_int(1),
    ),
    function_symbol="lunaflux_glm_hybrid_kda_decay_control_f32_v1",
    launch=try! @launch_contract.AotLaunchDimensions::new(
      grid_x=1024,
      grid_y=1,
      grid_z=1,
      block_x=256,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    operands=[
      kda_abi_operand(StepCountsInput, 20L, 4L),
      kda_abi_operand(KdaDecayRawForgetInput, 524288L, 16L),
      kda_abi_operand(bias_role, 32768L, 16L),
      kda_abi_operand(KdaDecayLogRateInput, 256L, 16L),
      kda_abi_operand(KdaDecayOutput, output_bytes, 16L),
    ],
  )
}

///|
test "KDA decay-control ABI authenticates F32 parameters and output" {
  let requirements = kda_abi_requirements()
  assert_true(
    exact_supported_launch_abi(
      requirements,
      kda_decay_control_claim(requirements),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kda_decay_control_claim(requirements, bias_role=KdaDecayLogRateInput),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kda_decay_control_claim(requirements, output_bytes=524288L),
    ),
  )
}
