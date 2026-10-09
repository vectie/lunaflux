///|
fn kv_assembly_abi_claim(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  suffix_role? : GlmHybridLaunchOperandRole = DsaAttentionKeyRopeInput,
  key_role? : GlmHybridLaunchOperandRole = DsaAttentionKeysInput,
  value_bytes? : Int64 = 12288L,
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
    operation=DsaKeyValueAttentionAssembly(8, 16, 8, 24),
    capability=DsaKeyValueAttentionAssemblyAot,
    module_digest=digest,
    module_bytes=b"abc",
    entry_point=@catalog.AotKernelEntryPoint::new(
      family~,
      entry_point_id=try! @catalog.AotKernelEntryPointId::from_stable_int(1),
    ),
    function_symbol="lunaflux_glm_hybrid_dsa_kv_assembly_rope_bf16_v1",
    launch=try! @launch_contract.AotLaunchDimensions::new(
      grid_x=48,
      grid_y=1,
      grid_z=1,
      block_x=256,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    operands=[
      kda_abi_operand(StepCountsInput, 20L, 4L),
      kda_abi_operand(DsaKeyValueHeadsOutput, 20480L, 16L),
      kda_abi_operand(suffix_role, 512L, 16L),
      kda_abi_operand(key_role, 12288L, 16L),
      kda_abi_operand(DsaAttentionValuesInput, value_bytes, 16L),
    ],
  )
}

///|
test "KV assembly admission fixes producer, suffix, and typed K/V outputs" {
  let requirements = kda_abi_requirements()
  assert_true(
    exact_supported_launch_abi(
      requirements,
      kv_assembly_abi_claim(requirements),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kv_assembly_abi_claim(requirements, suffix_role=DsaKeyRopeSuffixOutput),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kv_assembly_abi_claim(requirements, key_role=DsaAttentionValuesInput),
    ),
  )
  assert_false(
    exact_supported_launch_abi(
      requirements,
      kv_assembly_abi_claim(requirements, value_bytes=12286L),
    ),
  )
}
