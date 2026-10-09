///|
let operand_plan_test_table_bytes : Int = 129280 * 6 * 4

///|
fn operand_plan_test_requirements(
  model_id? : StringView = "deepseek-ai/DeepSeek-V4-Flash",
  fill? : Char = '1',
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements raise {
  let profile = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    model_id,
  )
  let metadata = @deepseek_v4_spec.DeepSeekV4ModelMetadata::from_verified_content(
    profile,
    @spec.ContentDigest::from_sha256(fill.to_string().repeat(64)),
  )
  @deepseek_v4_kernel_requirements.project(@deepseek_v4.build(metadata))
}

///|
fn operand_plan_test_hash_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is TokenHashRouting(_, _, _, _) {
      return ordinal
    }
  }
  abort("official token-hash requirement missing")
}

///|
fn operand_plan_test_target() -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn operand_plan_test_compiler() -> @luna_specialization.LunaCompilerPolicy {
  let identity = try! @luna_specialization.LunaCompilerIdentity::new(
    toolchain_digest_sha256=String::make(64, 'c'),
    major=12,
    minor=9,
    patch=1,
  )
  try! @luna_specialization.LunaCompilerPolicy::new(
    identity~,
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
}

///|
fn operand_plan_test_limits() -> @advanced_decoder_cuda_aot.AdvancedDecoderCudaAotLimits {
  try! @advanced_decoder_cuda_aot.AdvancedDecoderCudaAotLimits::new(
    maximum_tokens=4096,
    maximum_source_bytes=65536,
    maximum_recipe_bytes=65536,
  )
}

///|
fn operand_plan_test_candidate(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> @advanced_decoder_cuda_aot.AdvancedDecoderRoutingCudaCandidate {
  try! @advanced_decoder_cuda_aot.lower_token_hash_routing_cuda_aot_candidate(
    requirements,
    requirement_ordinal=operand_plan_test_hash_ordinal(requirements),
    maximum_tokens=2,
    target=operand_plan_test_target(),
    compiler=operand_plan_test_compiler(),
    limits=operand_plan_test_limits(),
  )
}

///|
fn operand_plan_test_source(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  released? : Bool = false,
) -> TokenHashSourceEvidence {
  let entries : Array[TokenHashTableEvidence] = []
  for layer in 0..<3 {
    entries.push({
      layer_index: layer,
      digest: String::make(64, ['d', 'e', 'f'][layer]),
      byte_count: operand_plan_test_table_bytes,
    })
  }
  {
    identity: requirements.source_identity(),
    source_manifest_digest: String::make(64, 'a'),
    tables: ReadOnlyArray::from_array(entries),
    released,
  }
}

///|
fn operand_plan_test_layout(
  identity : @spec.ModelIdentity,
  source_order? : ArrayView[Int] = [0, 1, 2],
) -> @segmented_device_materialize.SegmentedDeviceLayout {
  let regions : Array[
    @segmented_device_materialize.SegmentedDeviceTensorRegionInput,
  ] = []
  for ordinal, source_ordinal in source_order {
    regions.push(
      try! @segmented_device_materialize.SegmentedDeviceTensorRegionInput::new(
        tensor_ordinal=ordinal,
        source_arena_ordinal=source_ordinal,
        source_offset=0,
        byte_count=operand_plan_test_table_bytes,
      ),
    )
  }
  let limits = try! @segmented_device_materialize.SegmentedDeviceMaterializationLimits::new(
    alignment_bytes=256L,
    maximum_source_arenas=3,
    maximum_regions=3,
    maximum_host_arena_bytes=operand_plan_test_table_bytes,
    maximum_total_source_bytes=9308160L,
    maximum_device_arena_bytes=9308160L,
  )
  try! @segmented_device_materialize.plan_layout(
    source_identity=identity,
    source_digest=try! @segmented_device_materialize.SegmentedHostSourceDigest::from_sha256(
      String::make(64, 'b'),
    ),
    source_arena_lengths=[
      operand_plan_test_table_bytes, operand_plan_test_table_bytes, operand_plan_test_table_bytes,
    ],
    regions~,
    limits~,
  )
}

///|
fn operand_plan_test_candidate_evidence(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> TokenHashCandidateEvidence {
  try! candidate_evidence(operand_plan_test_candidate(requirements))
}
