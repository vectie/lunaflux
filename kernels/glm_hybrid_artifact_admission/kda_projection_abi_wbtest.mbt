///|
fn kda_projection_abi_claim(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  raw_role? : GlmHybridLaunchOperandRole = KdaRawForgetOutput,
  low_bytes? : Int64 = 8192L,
  gate_weight_bytes? : Int64 = 524288L,
  symbol? : String = "lunaflux_glm_hybrid_kda_projection_bf16_v1",
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
    operation=KdaProjection(64, 128),
    capability=KdaProjectionAot,
    module_digest=digest,
    module_bytes=b"abc",
    entry_point=entry,
    function_symbol=symbol,
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
      kda_abi_operand(HiddenInput, 262144L, 16L),
      kda_abi_operand(KdaQueryProjectionWeights, 67108864L, 16L),
      kda_abi_operand(KdaKeyProjectionWeights, 67108864L, 16L),
      kda_abi_operand(KdaValueProjectionWeights, 67108864L, 16L),
      kda_abi_operand(KdaForgetLowRankWeights, 1048576L, 16L),
      kda_abi_operand(KdaForgetExpansionWeights, 2097152L, 16L),
      kda_abi_operand(KdaInputGateWeights, gate_weight_bytes, 16L),
      kda_abi_operand(KdaProjectedQueryOutput, 524288L, 16L),
      kda_abi_operand(KdaProjectedKeyOutput, 524288L, 16L),
      kda_abi_operand(KdaProjectedValueOutput, 524288L, 16L),
      kda_abi_operand(KdaForgetLowRankOutput, low_bytes, 16L),
      kda_abi_operand(raw_role, 524288L, 16L),
      kda_abi_operand(KdaBetaOutput, 4096L, 16L),
    ],
  )
}

///|
test "KDA projection admission preserves raw forget and low-rank boundaries" {
  let requirements = kda_abi_requirements()
  assert_true(
    exact_supported_launch_abi(
      requirements,
      kda_projection_abi_claim(requirements),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kda_projection_abi_claim(requirements, raw_role=KdaLogDecayInput),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kda_projection_abi_claim(requirements, low_bytes=524288L),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kda_projection_abi_claim(requirements, gate_weight_bytes=524286L),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kda_projection_abi_claim(requirements, symbol="hostile_symbol"),
    ),
  )
}
