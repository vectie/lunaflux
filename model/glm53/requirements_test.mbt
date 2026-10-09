///|
fn full_metadata() -> @glm53_spec.Glm53FullMetadata {
  @glm53_spec.Glm53FullMetadata::from_verified_content(
    @glm53_spec.Glm53FullSpec::official(),
    try! @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
}

///|
fn flash_metadata() -> @glm53_spec.Glm53FlashMetadata {
  @glm53_spec.Glm53FlashMetadata::from_verified_content(
    @glm53_spec.Glm53FlashSpec::official(),
    try! @spec.ContentDigest::from_sha256(
      "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
}

///|
test "full profile fails closed on the dense backend capability set" {
  let dense = try! @family_contract.ModelBackendCapabilities::new([
    DenseCausalAttention,
  ])
  match @glm53.admit_full_backend(full_metadata(), dense) {
    MissingCapabilities(profile, missing) => {
      assert_eq(profile.architecture(), "GlmMoeDsaForCausalLM")
      assert_eq(profile.workload(), DecoderOnlyText)
      assert_eq(missing.length(), 3)
      assert_eq(missing[0], MixtureOfExperts)
      assert_eq(missing[1], MultiTokenPrediction)
      assert_eq(missing[2], DynamicSparseAttention)
    }
    SemanticCapabilitiesAvailable(_) =>
      fail("dense backend admitted full GLM-5.3")
  }
}

///|
test "Flash profile requires the complete hybrid multimodal capability set" {
  let partial = try! @family_contract.ModelBackendCapabilities::new([
    MixtureOfExperts,
    MultiTokenPrediction,
    HyperConnections,
    DynamicSparseAttention,
    VisionEncoder,
    MultimodalProjection,
  ])
  match @glm53.admit_flash_backend(flash_metadata(), partial) {
    MissingCapabilities(profile, missing) => {
      assert_eq(profile.architecture(), "Glm5NextForConditionalGeneration")
      assert_eq(profile.model_type(), "glm5_next")
      assert_eq(profile.workload(), MultimodalConditionalGeneration)
      assert_eq(missing.length(), 1)
      assert_eq(missing[0], LinearRecurrentAttention)
    }
    SemanticCapabilitiesAvailable(_) => fail("Flash backend omitted KDA")
  }
}

///|
test "complete Flash capability set is executable only after startup join" {
  let complete = try! @family_contract.ModelBackendCapabilities::new([
    MixtureOfExperts,
    MultiTokenPrediction,
    HyperConnections,
    DynamicSparseAttention,
    LinearRecurrentAttention,
    VisionEncoder,
    MultimodalProjection,
  ])
  assert_true(
    @glm53.admit_flash_backend(flash_metadata(), complete)
    is SemanticCapabilitiesAvailable(_),
  )
}
