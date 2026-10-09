///|
fn kda_gated_norm_claim(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  gate_role? : GlmHybridLaunchOperandRole = KdaGatedNormGateInput,
  epsilon? : Double = 0.00001,
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
    operation=KdaGatedRmsNorm(64, 128, epsilon),
    capability=KdaGatedRmsNormAot,
    module_digest=digest,
    module_bytes=b"abc",
    entry_point=@catalog.AotKernelEntryPoint::new(
      family~,
      entry_point_id=try! @catalog.AotKernelEntryPointId::from_stable_int(1),
    ),
    function_symbol="lunaflux_glm_hybrid_kda_gated_rms_norm_bf16_v1",
    launch=try! @launch_contract.AotLaunchDimensions::new(
      grid_x=1,
      grid_y=1,
      grid_z=1,
      block_x=1,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    operands=[
      kda_abi_operand(StepCountsInput, 20L, 4L),
      kda_abi_operand(KdaGatedNormRecurrentInput, 524288L, 16L),
      kda_abi_operand(gate_role, 524288L, 16L),
      kda_abi_operand(KdaGatedNormWeights, 256L, 16L),
      kda_abi_operand(KdaGatedNormOutput, 524288L, 16L),
    ],
  )
}

///|
test "gated RMSNorm ABI admits exact roles and rejects gate substitution" {
  let requirements = kda_abi_requirements()
  assert_true(
    exact_supported_launch_abi(requirements, kda_gated_norm_claim(requirements)),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kda_gated_norm_claim(requirements, gate_role=KdaOutputGateOutput),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kda_gated_norm_claim(requirements, epsilon=0.00002),
    ),
  )
}
