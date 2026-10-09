///|
fn kda_output_projection_claim(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  weight_role? : GlmHybridLaunchOperandRole = KdaOutputProjectionWeights,
  output_bytes? : Int64 = 262144L,
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
    operation=KdaOutputProjection(8192, 4096),
    capability=KdaOutputProjectionAot,
    module_digest=digest,
    module_bytes=b"abc",
    entry_point=@catalog.AotKernelEntryPoint::new(
      family~,
      entry_point_id=try! @catalog.AotKernelEntryPointId::from_stable_int(1),
    ),
    function_symbol="lunaflux_glm_hybrid_kda_output_projection_bf16_v1",
    launch=try! @launch_contract.AotLaunchDimensions::new(
      grid_x=512,
      grid_y=1,
      grid_z=1,
      block_x=256,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    operands=[
      kda_abi_operand(StepCountsInput, 20L, 4L),
      kda_abi_operand(KdaOutputProjectionInput, 524288L, 16L),
      kda_abi_operand(weight_role, 67108864L, 16L),
      kda_abi_operand(KdaOutputProjectionOutput, output_bytes, 16L),
    ],
  )
}

///|
test "KDA output projection ABI authenticates exact weights and output shape" {
  let requirements = kda_abi_requirements()
  assert_true(
    exact_supported_launch_abi(
      requirements,
      kda_output_projection_claim(requirements),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kda_output_projection_claim(requirements, weight_role=ModelWeights),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kda_output_projection_claim(requirements, output_bytes=262142L),
    ),
  )
}
