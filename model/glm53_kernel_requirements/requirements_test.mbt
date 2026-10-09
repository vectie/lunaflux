///|
fn kernel_full_metadata(fill? : Char = 'c') -> @glm53_spec.Glm53FullMetadata {
  @glm53_spec.Glm53FullMetadata::from_verified_content(
    @glm53_spec.Glm53FullSpec::official(),
    try! @spec.ContentDigest::from_sha256(fill.to_string().repeat(64)),
  )
}

///|
fn kernel_flash_metadata(fill? : Char = 'd') -> @glm53_spec.Glm53FlashMetadata {
  @glm53_spec.Glm53FlashMetadata::from_verified_content(
    @glm53_spec.Glm53FlashSpec::official(),
    try! @spec.ContentDigest::from_sha256(fill.to_string().repeat(64)),
  )
}

///|
test "full requirements include DSA compute reuse MoE and MTP" {
  let requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    kernel_full_metadata(),
  )
  assert_eq(requirements.requirements().length(), 28)
  assert_eq(requirements.capabilities().length(), 26)
  assert_eq(requirements.geometry().vocabulary_size(), 154880)
  assert_eq(requirements.geometry().hidden_size(), 6144)
  assert_eq(requirements.geometry().decoder_layer_count(), 78)
  assert_eq(requirements.geometry().context_length(), 1048576)
  let operations = requirements.requirements().map(r => r.operation())
  assert_true(operations.contains(DsaQueryAProjection(6144, 2048)))
  assert_true(operations.contains(DsaKeyValueAProjection(6144, 512, 64)))
  assert_true(operations.contains(DsaQueryARmsNorm(2048, 0.00001)))
  assert_true(operations.contains(DsaQueryBProjection(2048, 64, 256)))
  assert_true(
    operations.contains(DsaKeyValueAttentionAssembly(64, 192, 64, 256)),
  )
  assert_true(requirements.capabilities().contains(DsaQueryAProjectionAot))
  assert_true(requirements.capabilities().contains(DsaKeyValueAProjectionAot))
  assert_true(requirements.capabilities().contains(DsaQueryARmsNormAot))
  assert_true(requirements.capabilities().contains(DsaQueryBProjectionAot))
  assert_true(operations.contains(DsaIndexCompute(32, 128, 2048, true)))
  assert_true(operations.contains(DsaIndexReuse))
  assert_true(operations.contains(MoeSigmoidAndCorrectionBias(256)))
  assert_true(
    operations.contains(
      MoeGroupedNoAuxiliaryLossTopK(256, 1, 2, 1, 8, true, 2.5),
    ),
  )
  assert_true(
    requirements.capabilities().contains(MoeSigmoidAndCorrectionBiasAot),
  )
  assert_false(operations.contains(ExpertScoreTransform(Sigmoid)))
  assert_true(operations.contains(MultiTokenPredictionInputProjection))
  assert_true(operations.contains(MoeDecoderResidualAdd(6144)))
  assert_true(requirements.capabilities().contains(MoeDecoderResidualAddAot))
  assert_true(operations.contains(AuxiliaryDecoderStage(0)))
  assert_true(operations.contains(FinalRmsNorm(6144, 0.00001)))
  assert_false(operations.contains(KdaProjection(64, 128)))
  assert_false(operations.contains(VisionRmsNorm(1024)))
}

///|
test "Flash requirements include pooling KDA mHC and complete vision stages" {
  let requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    kernel_flash_metadata(),
  )
  assert_eq(requirements.requirements().length(), 47)
  assert_eq(requirements.capabilities().length(), 44)
  assert_eq(requirements.geometry().vocabulary_size(), 154880)
  assert_eq(requirements.geometry().hidden_size(), 4096)
  let operations = requirements.requirements().map(r => r.operation())
  assert_true(operations.contains(DsaQueryAProjection(4096, 1536)))
  assert_true(operations.contains(DsaKeyValueAProjection(4096, 512, 0)))
  assert_true(operations.contains(DsaQueryARmsNorm(1536, 0.00001)))
  assert_true(operations.contains(DsaQueryBProjection(1536, 64, 256)))
  assert_true(
    operations.contains(DsaKeyValueAttentionAssembly(64, 256, 0, 256)),
  )
  assert_true(requirements.capabilities().contains(DsaQueryAProjectionAot))
  assert_true(requirements.capabilities().contains(DsaKeyValueAProjectionAot))
  assert_true(requirements.capabilities().contains(DsaQueryARmsNormAot))
  assert_true(requirements.capabilities().contains(DsaQueryBProjectionAot))
  assert_true(operations.contains(DsaIndexPooling(4, true, true)))
  assert_true(operations.contains(KdaProjection(64, 128)))
  assert_true(operations.contains(KdaShortConvolution(8192, 4)))
  assert_true(operations.contains(KdaDecayControl(64, 128, -5.0)))
  assert_true(requirements.capabilities().contains(KdaDecayControlAot))
  assert_true(operations.contains(KdaOutputGateProjection(4096, 64, 128)))
  assert_true(operations.contains(KdaGatedRmsNorm(64, 128, 0.00001)))
  assert_true(operations.contains(MoeSigmoidAndCorrectionBias(288)))
  assert_true(
    operations.contains(
      MoeGroupedNoAuxiliaryLossTopK(288, 1, 2, 1, 8, true, 2.5),
    ),
  )
  assert_true(
    requirements.capabilities().contains(MoeSigmoidAndCorrectionBiasAot),
  )
  assert_false(operations.contains(ExpertScoreTransform(Sigmoid)))
  assert_true(requirements.capabilities().contains(KdaOutputGateProjectionAot))
  assert_true(requirements.capabilities().contains(KdaGatedRmsNormAot))
  assert_true(
    operations.contains(HyperConnectionFunction(4, 4096, 0.000001, 0.00001)),
  )
  assert_true(operations.contains(HyperConnectionSinkhorn(4, 20, 0.000001)))
  assert_true(operations.contains(HyperConnectionMix(4)))
  assert_true(
    requirements.capabilities().contains(Advanced(HyperConnectionMixAot)),
  )
  assert_true(operations.contains(VisionPatchProjection(2, 14, 3, 1024)))
  assert_true(operations.contains(VisionAttention(16, 64)))
  assert_true(operations.contains(VisionSpatialMerge(1024, 4096, 2)))
  assert_true(operations.contains(VisionDecoderProjection(4096, 4096)))
  assert_true(operations.contains(VisionPostProjectionNorm(4096)))
  assert_true(operations.contains(FinalRmsNorm(4096, 0.00001)))
}

///|
test "requirement identity is deterministic and content bound" {
  let first = @glm53_kernel_requirements.build_flash_kernel_requirements(
    kernel_flash_metadata(),
  )
  let same = @glm53_kernel_requirements.build_flash_kernel_requirements(
    kernel_flash_metadata(),
  )
  let changed = @glm53_kernel_requirements.build_flash_kernel_requirements(
    kernel_flash_metadata(fill='e'),
  )
  assert_eq(first, same)
  assert_true(first.source_identity() != changed.source_identity())
  assert_true(first.digest() != changed.digest())
  assert_eq(first.digest().as_hex().length(), 64)
}

///|
test "generic projection fails at first unsupported hybrid requirement" {
  let requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    kernel_full_metadata(),
  )
  assert_eq(
    requirements.requirements()[0].current_generic_capability(ordinal=0),
    @plan.KernelCapabilityId::embedding_lookup(),
  )
  assert_eq(
    requirements.requirements()[1].current_generic_capability(ordinal=1),
    @plan.KernelCapabilityId::rms_norm(),
  )
  try requirements.require_current_generic_capabilities() catch {
    UnsupportedGenericKernelOperation(2, DsaQueryAProjection(6144, 2048)) => ()
    _ => fail("generic projection rejected the wrong operation")
  } noraise {
    _ => fail("hybrid semantics were weakened to generic kernel IDs")
  }
}

///|
test "hostile bounds fail before a requirement set escapes" {
  try
    @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementLimits::new(
      maximum_layers=0,
      maximum_vision_layers=1,
      maximum_requirements=1,
    )
  catch {
    InvalidLimit => ()
    _ => fail("invalid limits returned the wrong error")
  } noraise {
    _ => fail("zero layer limit admitted")
  }
  let limits = @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementLimits::new(
    maximum_layers=8,
    maximum_vision_layers=8,
    maximum_requirements=8,
  )
  try
    @glm_hybrid_kernel_requirements.derive_requirements(
      @glm53.build_full_execution_plan(kernel_full_metadata()),
      limits~,
    )
  catch {
    PlanExceedsLimits => ()
    _ => fail("bounded plan returned the wrong error")
  } noraise {
    _ => fail("oversized plan admitted")
  }
}
