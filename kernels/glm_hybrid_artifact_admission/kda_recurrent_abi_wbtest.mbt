///|
fn kda_recurrent_abi_claim(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  operation? : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation = KdaRecurrentUpdate(
    64, 128, -5.0,
  ),
  beta_role? : GlmHybridLaunchOperandRole = KdaInputGateInput,
  beta_bytes? : Int64 = 4096L,
  state_bytes? : Int64 = 33554432L,
  offsets_bytes? : Int64 = 36L,
  symbol? : String = "lunaflux_glm_hybrid_kda_recurrent_delta_bf16_f32_v1",
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
    grid_x=1,
    grid_y=1,
    grid_z=1,
    block_x=1,
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
    capability=KdaRecurrentUpdateAot,
    module_digest=digest,
    module_bytes=b"abc",
    entry_point=entry,
    function_symbol=symbol,
    launch~,
    operands=[
      kda_abi_operand(StepCountsInput, 20L, 4L),
      kda_abi_operand(KdaSequenceOffsetsInput, offsets_bytes, 4L),
      kda_abi_operand(KdaRecurrentQueryInput, 524288L, 16L),
      kda_abi_operand(KdaRecurrentKeyInput, 524288L, 16L),
      kda_abi_operand(KdaRecurrentValueInput, 524288L, 16L),
      kda_abi_operand(KdaLogDecayInput, 1048576L, 16L),
      kda_abi_operand(beta_role, beta_bytes, 16L),
      kda_abi_operand(KdaRecurrentInitialStateInput, state_bytes, 16L),
      kda_abi_operand(KdaRecurrentOutput, 524288L, 16L),
      kda_abi_operand(KdaRecurrentFinalStateOutput, 33554432L, 16L),
    ],
  )
}

///|
test "KDA recurrent admission authenticates mixed state ABI and hostile mismatches" {
  let requirements = kda_abi_requirements()
  assert_true(
    exact_supported_launch_abi(
      requirements,
      kda_recurrent_abi_claim(requirements),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kda_recurrent_abi_claim(requirements, beta_role=HiddenInput),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kda_recurrent_abi_claim(requirements, beta_bytes=8192L),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kda_recurrent_abi_claim(requirements, state_bytes=33554428L),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kda_recurrent_abi_claim(requirements, offsets_bytes=32L),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kda_recurrent_abi_claim(
        requirements,
        operation=KdaRecurrentUpdate(64, 128, -4.0),
      ),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kda_recurrent_abi_claim(requirements, symbol="hostile_symbol"),
    ),
  )
}
