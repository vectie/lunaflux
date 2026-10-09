///|
fn moe_feed_forward_claim(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  output_role? : GlmHybridLaunchOperandRole = MoeRoutedOutput,
  selected_bytes? : Int64 = 256L,
  symbol? : String = "lunaflux_glm_hybrid_moe_routed_experts_bf16_v1",
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
    operation=RoutedExpertFeedForward(288, 16, SwiGlu),
    capability=Advanced(RoutedExpertFeedForwardAot),
    module_digest=digest,
    module_bytes=b"abc",
    entry_point=@catalog.AotKernelEntryPoint::new(
      family~,
      entry_point_id=try! @catalog.AotKernelEntryPointId::from_stable_int(1),
    ),
    function_symbol=symbol,
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
      kda_abi_operand(MoeFeedForwardInput, 262144L, 16L),
      kda_abi_operand(MoeSelectedIndicesOutput, selected_bytes, 4L),
      kda_abi_operand(MoeSelectedWeightsOutput, 256L, 16L),
      kda_abi_operand(MoeExpertGateWeights, 37748736L, 16L),
      kda_abi_operand(MoeExpertUpWeights, 37748736L, 16L),
      kda_abi_operand(MoeExpertDownWeights, 37748736L, 16L),
      kda_abi_operand(output_role, 262144L, 16L),
    ],
  )
}

///|
test "MoE routed expert admission authenticates selected width output role and symbol" {
  let requirements = kda_abi_requirements(routed_expert_count=288)
  assert_true(
    exact_supported_launch_abi(
      requirements,
      moe_feed_forward_claim(requirements),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      moe_feed_forward_claim(requirements, selected_bytes=252L),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      moe_feed_forward_claim(requirements, output_role=MoeFeedForwardInput),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      moe_feed_forward_claim(requirements, symbol="wrong_symbol"),
    ),
  )
}
