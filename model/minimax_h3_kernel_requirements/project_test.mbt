///|
fn minimax_kernel_metadata(
  variant : @minimax_h3_spec.MiniMaxH3Variant,
  seed? : Char = '7',
) -> @minimax_h3_spec.MiniMaxH3ModelMetadata {
  @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
    @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
      variant~,
      checkpoint_format=DiffusersV1,
    ),
    try! @spec.ContentDigest::from_sha256(String::make(64, seed)),
  )
}

///|
test "FL2VA projects exact ordered joint diffusion AOT semantics" {
  let requirements = @minimax_h3_kernel_requirements.project(
    minimax_kernel_metadata(Fl2va),
  )
  assert_eq(requirements.requirements().map(entry => entry.operation()), [
    TokenTextConditioning(5120),
    OrderedKeyframeConditioning(2, 24),
    VideoLatentProjection(24, 1, 2, 2, 5376),
    AudioLatentProjection(32, 5376),
    JointTransformerAttention(50, 2, 56, 128, 5376),
    JointTransformerMlp(50, 2, 5376, 14336),
    VideoFinalRmsNorm(5376, 1, 100000),
    AudioFinalRmsNorm(5376, 1, 100000),
    VideoFinalAdaLnF32Boundary(5376),
    AudioFinalAdaLnF32Boundary(5376),
    FinalAdaLnParameterRow(2688, 5376),
    PackedModalityAdaLnRows(5376),
    TimestepFrequencyRows(256),
    TimestepEmbeddingMlpRows(256, 5376, 2688),
    DistinctTimestepAdaLnParameterRows(2688, 5376),
    VideoOutputProjection(24, 1, 2, 2, 5376),
    AudioOutputProjection(32, 5376),
    ShiftedFlowStep(VideoLatent, 12, 1, 1000),
    ShiftedFlowStep(AudioLatent, 3, 1, 1000),
    VideoVaeLatentDestandardization(
      24,
      Float32InputScaleMeanOutput,
      BatchChannelFrameHeightWidth,
    ),
    VideoVaeDecoderBody(24, 16, 17, 5),
    AudioVaeLatentDestandardization(
      32,
      2,
      Float32InputScaleMeanOutput,
      BatchChannelTime,
    ),
    AudioVaeDecoderInputProjection(
      2,
      32,
      2048,
      1,
      Float32InputWeightBiasOutput,
      BatchChannelTime,
    ),
    AudioVaeDecoderConvPre(
      2,
      2048,
      1024,
      7,
      3,
      1,
      Float32InputWeightGWeightVBiasWorkspaceOutput,
      PerOutputChannelInputAndKernel,
      BatchChannelTime,
    ),
    AudioVaeDecoderFirstUpsample(
      2,
      1024,
      512,
      9,
      5,
      2,
      Float32InputWeightGWeightVBiasWorkspaceOutput,
      PerInputChannelOutputAndKernel,
      BatchChannelTime,
    ),
    AudioVaeDecoderFirstAmpActivation(
      2,
      512,
      2,
      12,
      Float32InputAlphaBetaUpsampleFilterDownsampleFilterWorkspacesOutput,
      BatchChannelTime,
    ),
    AudioVaeDecoderFirstAmpStage(
      2,
      512,
      3,
      3,
      Float32OriginalPrecomputedActivationPackedParametersWorkspaceOutput,
      BatchChannelTime,
    ),
    AudioVaeDecoderUpsampleTransition(
      1,
      2,
      512,
      256,
      9,
      5,
      2,
      Float32InputWeightGWeightVBiasWorkspaceOutput,
      PerInputChannelOutputAndKernel,
      BatchChannelTime,
    ),
    AudioVaeDecoderAmpActivation(
      1,
      2,
      256,
      2,
      12,
      Float32InputAlphaBetaUpsampleFilterDownsampleFilterWorkspacesOutput,
      BatchChannelTime,
    ),
    AudioVaeDecoderAmpStage(
      1,
      2,
      256,
      3,
      3,
      Kernels3_7_11Dilations1_3_5,
      Float32OriginalPrecomputedActivationPackedParametersWorkspaceOutput,
      BatchChannelTime,
    ),
    AudioVaeDecoderUpsampleTransition(
      2,
      2,
      256,
      128,
      4,
      2,
      1,
      Float32InputWeightGWeightVBiasWorkspaceOutput,
      PerInputChannelOutputAndKernel,
      BatchChannelTime,
    ),
    AudioVaeDecoderAmpActivation(
      2,
      2,
      128,
      2,
      12,
      Float32InputAlphaBetaUpsampleFilterDownsampleFilterWorkspacesOutput,
      BatchChannelTime,
    ),
    AudioVaeDecoderBody(128, 40, 32000, 2),
  ])
  assert_eq(requirements.capabilities(), [
    TokenTextConditioningAot,
    OrderedKeyframeConditioningAot,
    VideoLatentProjectionAot,
    AudioLatentProjectionAot,
    JointTransformerAttentionAot,
    JointTransformerMlpAot,
    VideoFinalRmsNormAot,
    AudioFinalRmsNormAot,
    VideoFinalAdaLnF32BoundaryAot,
    AudioFinalAdaLnF32BoundaryAot,
    FinalAdaLnParameterRowAot,
    PackedModalityAdaLnRowsAot,
    TimestepFrequencyRowsAot,
    TimestepEmbeddingMlpRowsAot,
    DistinctTimestepAdaLnParameterRowsAot,
    VideoOutputProjectionAot,
    AudioOutputProjectionAot,
    ShiftedFlowStepAot,
    VideoVaeLatentDestandardizationAot,
    VideoVaeDecoderBodyAot,
    AudioVaeLatentDestandardizationAot,
    AudioVaeDecoderInputProjectionAot,
    AudioVaeDecoderConvPreAot,
    AudioVaeDecoderFirstUpsampleAot,
    AudioVaeDecoderFirstAmpActivationAot,
    AudioVaeDecoderFirstAmpStageAot,
    AudioVaeDecoderUpsampleTransitionAot,
    AudioVaeDecoderAmpActivationAot,
    AudioVaeDecoderAmpStageAot,
    AudioVaeDecoderBodyAot,
  ])
  assert_eq(
    requirements.projection_contract().storage(),
    Float32InputWeightBiasBFloat16Output,
  )
  assert_eq(
    requirements.projection_contract().output_storage(),
    Float32InputWeightBiasFloat32Output,
  )
  assert_eq(
    requirements.projection_contract().video_layout(),
    BatchFrameHeightWidthChannelTemporalHeightWidth,
  )
}

///|
test "Ref2VA projection retains exact ordered reference ceilings" {
  let requirements = @minimax_h3_kernel_requirements.project(
    minimax_kernel_metadata(Ref2va),
  )
  assert_eq(
    requirements.requirements()[0].operation(),
    TokenTextConditioning(5120),
  )
  assert_eq(
    requirements.requirements()[1].operation(),
    OrderedReferenceConditioning(9, 3, 3, 12, false),
  )
  assert_eq(
    requirements.requirements()[1].capability(),
    OrderedReferenceConditioningAot,
  )
  assert_eq(requirements.requirements().length(), 33)
  assert_eq(requirements.capabilities().length(), 30)
}

///|
test "MiniMax projection digest is deterministic variant and content sensitive" {
  let fl = @minimax_h3_kernel_requirements.project(
    minimax_kernel_metadata(Fl2va),
  )
  let repeated = @minimax_h3_kernel_requirements.project(
    minimax_kernel_metadata(Fl2va),
  )
  let reference = @minimax_h3_kernel_requirements.project(
    minimax_kernel_metadata(Ref2va),
  )
  let other_content = @minimax_h3_kernel_requirements.project(
    minimax_kernel_metadata(Fl2va, seed='8'),
  )
  assert_eq(fl.digest(), repeated.digest())
  assert_true(fl.source_identity() != reference.source_identity())
  assert_true(fl.digest() != reference.digest())
  assert_true(fl.source_identity() != other_content.source_identity())
  assert_true(fl.digest() != other_content.digest())
}

///|
test "MiniMax requirements never alias current generic decoder kernels" {
  let variants : Array[@minimax_h3_spec.MiniMaxH3Variant] = [Fl2va, Ref2va]
  for variant in variants {
    let requirements = @minimax_h3_kernel_requirements.project(
      minimax_kernel_metadata(variant),
    )
    try requirements.require_current_generic_decoder_capabilities() catch {
      UnsupportedGenericDecoderKernelOperation(0, TokenTextConditioning(5120)) =>
        ()
      _ => fail("MiniMax returned the wrong decoder-kernel rejection")
    } noraise {
      _ => fail("MiniMax diffusion work was weakened into decoder kernels")
    }
  }
}
