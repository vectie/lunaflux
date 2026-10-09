///|
fn hyper_mix_claim(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  post_role? : GlmHybridLaunchOperandRole = HyperPostControlInput,
  output_role? : GlmHybridLaunchOperandRole = HyperMixedStreamsOutput,
  combine_bytes? : Int64 = 2048L,
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
    operation=HyperConnectionMix(4),
    capability=Advanced(HyperConnectionMixAot),
    module_digest=digest,
    module_bytes=b"abc",
    entry_point=@catalog.AotKernelEntryPoint::new(
      family~,
      entry_point_id=try! @catalog.AotKernelEntryPointId::from_stable_int(1),
    ),
    function_symbol="lunaflux_glm_hybrid_hyper_connection_mix_bf16_v1",
    launch=try! @launch_contract.AotLaunchDimensions::new(
      grid_x=2048,
      grid_y=1,
      grid_z=1,
      block_x=256,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    operands=[
      kda_abi_operand(StepCountsInput, 20L, 4L),
      kda_abi_operand(post_role, 512L, 16L),
      kda_abi_operand(HyperCombineControlInput, combine_bytes, 16L),
      kda_abi_operand(HyperResidualStreamsInput, 1048576L, 16L),
      kda_abi_operand(HyperSublayerOutputInput, 262144L, 16L),
      kda_abi_operand(output_role, 1048576L, 16L),
    ],
  )
}

///|
test "hyper mix admission fixes F32 controls, distinct output, and extents" {
  let requirements = kda_abi_requirements(
    routed_expert_count=288,
    hyper_connection=true,
  )
  assert_true(
    exact_supported_launch_abi(requirements, hyper_mix_claim(requirements)),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      hyper_mix_claim(requirements, post_role=HyperCombineControlInput),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      hyper_mix_claim(requirements, output_role=HyperResidualStreamsInput),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      hyper_mix_claim(requirements, combine_bytes=2044L),
    ),
  )
}
