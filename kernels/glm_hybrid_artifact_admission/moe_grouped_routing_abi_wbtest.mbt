///|
fn grouped_routing_claim(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  operation? : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation = MoeGroupedNoAuxiliaryLossTopK(
    288, 1, 2, 1, 8, true, 2.5,
  ),
  choice_role? : GlmHybridLaunchOperandRole = MoeChoiceScoresOutput,
  index_bytes? : Int64 = 1024L,
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
    operation~,
    capability=MoeGroupedRoutingAot,
    module_digest=digest,
    module_bytes=b"abc",
    entry_point=@catalog.AotKernelEntryPoint::new(
      family~,
      entry_point_id=try! @catalog.AotKernelEntryPointId::from_stable_int(1),
    ),
    function_symbol="lunaflux_glm_hybrid_moe_grouped_routing_f32_v1",
    launch=try! @launch_contract.AotLaunchDimensions::new(
      grid_x=32,
      grid_y=1,
      grid_z=1,
      block_x=1,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    operands=[
      kda_abi_operand(StepCountsInput, 20L, 4L),
      kda_abi_operand(MoeUncorrectedScoresOutput, 36864L, 16L),
      kda_abi_operand(choice_role, 36864L, 16L),
      kda_abi_operand(MoeSelectedIndicesOutput, index_bytes, 4L),
      kda_abi_operand(MoeSelectedWeightsOutput, 1024L, 16L),
    ],
  )
}

///|
test "grouped routing admission authenticates streams geometry and outputs" {
  let requirements = kda_abi_requirements(routed_expert_count=288)
  assert_true(
    exact_supported_launch_abi(
      requirements,
      grouped_routing_claim(requirements),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      grouped_routing_claim(
        requirements,
        choice_role=MoeUncorrectedScoresOutput,
      ),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      grouped_routing_claim(requirements, index_bytes=1020L),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      grouped_routing_claim(
        requirements,
        operation=MoeGroupedNoAuxiliaryLossTopK(288, 5, 2, 1, 8, true, 2.5),
      ),
    ),
  )
}
