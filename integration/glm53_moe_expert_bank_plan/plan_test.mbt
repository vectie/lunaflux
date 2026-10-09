///|
fn bank_test_full_metadata(
  digest? : String = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
) -> @glm53_spec.Glm53FullMetadata {
  @glm53_spec.Glm53FullMetadata::from_verified_content(
    @glm53_spec.Glm53FullSpec::official(),
    try! @spec.ContentDigest::from_sha256(digest),
  )
}

///|
fn bank_test_flash_metadata() -> @glm53_spec.Glm53FlashMetadata {
  @glm53_spec.Glm53FlashMetadata::from_verified_content(
    @glm53_spec.Glm53FlashSpec::official(),
    try! @spec.ContentDigest::from_sha256(
      "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    ),
  )
}

///|
fn bank_test_target() -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn bank_test_compiler() -> @luna_specialization.LunaCompilerPolicy {
  try! @luna_specialization.LunaCompilerPolicy::new(
    identity=try! @luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256=String::make(64, 'c'),
      major=12,
      minor=9,
      patch=1,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
}

///|
fn bank_test_limits() -> @glm_hybrid_cuda_aot.GlmHybridCudaAotLimits {
  try! @glm_hybrid_cuda_aot.GlmHybridCudaAotLimits::new(
    maximum_tokens=4096,
    maximum_source_bytes=65536,
    maximum_recipe_bytes=65536,
  )
}

///|
fn bank_test_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  shared : Bool,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match (shared, requirement.operation()) {
      (false, RoutedExpertFeedForward(_, _, _)) => return ordinal
      (true, SharedExpertFeedForward(_, _, _)) => return ordinal
      _ => ()
    }
  }
  abort("missing expert requirement")
}

///|
fn bank_test_candidates(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> (
  @glm_hybrid_cuda_aot.GlmHybridMoeFeedForwardCudaCandidate,
  @glm_hybrid_cuda_aot.GlmHybridMoeFeedForwardCudaCandidate,
) {
  (
    try! @glm_hybrid_cuda_aot.lower_moe_routed_expert_feed_forward_cuda_aot_candidate(
      requirements,
      requirement_ordinal=bank_test_ordinal(requirements, false),
      maximum_tokens=32,
      target=bank_test_target(),
      compiler=bank_test_compiler(),
      limits=bank_test_limits(),
    ),
    try! @glm_hybrid_cuda_aot.lower_moe_shared_expert_feed_forward_cuda_aot_candidate(
      requirements,
      requirement_ordinal=bank_test_ordinal(requirements, true),
      maximum_tokens=32,
      target=bank_test_target(),
      compiler=bank_test_compiler(),
      limits=bank_test_limits(),
    ),
  )
}

///|
test "Full and Flash expert-bank plans bind manifest layouts to exact candidates" {
  let full_metadata = bank_test_full_metadata()
  let flash_metadata = bank_test_flash_metadata()
  let full_manifest = @glm53_weights.build_full_tensor_manifest(
    full_metadata,
    BFloat16WithFloat32Routing,
  )
  let flash_manifest = @glm53_weights.build_flash_tensor_manifest(
    flash_metadata,
    BFloat16WithFloat32Routing,
  )
  let full_requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata,
  )
  let flash_requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    flash_metadata,
  )
  let (full_routed, full_shared) = bank_test_candidates(full_requirements)
  let (flash_routed, flash_shared) = bank_test_candidates(flash_requirements)
  let full_layout = @glm53_weights.build_moe_expert_bank_layout(
    full_manifest,
    layer_index=3,
  )
  let flash_layout = @glm53_weights.build_moe_expert_bank_layout(
    flash_manifest,
    layer_index=3,
  )
  let full = join_moe_expert_bank_candidates(
    full_manifest,
    full_layout,
    full_routed,
    full_shared,
    expected_layer_index=3,
  )
  let flash = join_moe_expert_bank_candidates(
    flash_manifest,
    flash_layout,
    flash_routed,
    flash_shared,
    expected_layer_index=3,
  )
  assert_eq(full.execution_identity(), full_manifest.execution_identity())
  assert_eq(full.layer_index(), 3)
  assert_true(full.digest().as_hex() != flash.digest().as_hex())
  assert_false(full.materialization_authority())
  assert_false(full.upload_authority())
  assert_false(full.launch_authority())
  inspect(
    full_layout.digest().as_hex() +
    ":" +
    flash_layout.digest().as_hex() +
    ":" +
    full.digest().as_hex() +
    ":" +
    flash.digest().as_hex(),
    content="34f559cf4c8beb8ebef46c06481598f844aa9e3ed7112a80d4b09d1db6af5f9e:8360f2fcd0ce94313437bc127541e39c6457c5a5394b2cc02a10b3faea98885f:66b46c0ed8c0e80e0b087de5fc4a7fc4e622df3579c71842170a204a6a981994:323d264c8e4a01c7f71744fd7d7ab4ecd5816287cc96854468f4c559851687b6",
  )
}

///|
test "expert-bank join rejects wrong layer manifest identity-digest and profile" {
  let metadata = bank_test_full_metadata()
  let manifest = @glm53_weights.build_full_tensor_manifest(
    metadata,
    BFloat16WithFloat32Routing,
  )
  let requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    metadata,
  )
  let (routed, shared) = bank_test_candidates(requirements)
  let layout = @glm53_weights.build_moe_expert_bank_layout(
    manifest,
    layer_index=3,
  )
  try
    join_moe_expert_bank_candidates(
      manifest,
      layout,
      routed,
      shared,
      expected_layer_index=4,
    )
  catch {
    LayerMismatch => ()
    _ => fail("wrong layer error")
  } noraise {
    _ => fail("wrong layer admitted")
  }
  let foreign_manifest = @glm53_weights.build_full_tensor_manifest(
    bank_test_full_metadata(
      digest="dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
    ),
    BFloat16WithFloat32Routing,
  )
  try
    join_moe_expert_bank_candidates(
      foreign_manifest,
      layout,
      routed,
      shared,
      expected_layer_index=3,
    )
  catch {
    IdentityMismatch | ManifestDigestMismatch => ()
    _ => fail("wrong manifest identity/digest error")
  } noraise {
    _ => fail("foreign manifest identity/digest admitted")
  }
  let flash_metadata = bank_test_flash_metadata()
  let flash_manifest = @glm53_weights.build_flash_tensor_manifest(
    flash_metadata,
    BFloat16WithFloat32Routing,
  )
  try
    join_moe_expert_bank_candidates(
      flash_manifest,
      layout,
      routed,
      shared,
      expected_layer_index=3,
    )
  catch {
    IdentityMismatch | ProfileMismatch => ()
    _ => fail("wrong profile error")
  } noraise {
    _ => fail("wrong profile admitted")
  }
}
