///|
/// Bounded startup failures. No artifact, prompt, or request payload is
/// retained by these errors.
pub(all) suberror JointDiffusionKernelRequirementsError {
  InvalidLimit
  UnsupportedSchema
  InvalidSourceIdentity
  PlanExceedsLimits
  DuplicateSemanticOperation(Int32)
  RequirementsEncodingTooLarge
  UnsupportedGenericDecoderKernelOperation(
    Int32,
    JointDiffusionSemanticOperation
  )
} derive(Debug, Eq)

///|
/// Explicit derivation ceilings. Requirement derivation is startup-only work.
pub struct JointDiffusionKernelRequirementLimits {
  priv maximum_transformer_layers : Int32
  priv maximum_requirements : Int32
} derive(Debug, Eq)

///|
pub fn JointDiffusionKernelRequirementLimits::new(
  maximum_transformer_layers~ : Int32,
  maximum_requirements~ : Int32,
) -> Self raise JointDiffusionKernelRequirementsError {
  if maximum_transformer_layers <= 0 ||
    maximum_transformer_layers > 4096 ||
    maximum_requirements <= 0 ||
    maximum_requirements > 64 {
    raise InvalidLimit
  }
  { maximum_transformer_layers, maximum_requirements, }
}

///|
/// Exact family-neutral work implied by a joint audio/video diffusion plan.
/// Geometry that changes AOT specialization remains part of each operation.
pub(all) enum JointDiffusionSemanticOperation {
  TokenTextConditioning(Int32)
  OrderedKeyframeConditioning(Int32, Int32)
  OrderedReferenceConditioning(Int32, Int32, Int32, Int32, Bool)
  VideoLatentProjection(Int32, Int32, Int32, Int32, Int32)
  AudioLatentProjection(Int32, Int32)
  JointTransformerAttention(Int32, Int32, Int32, Int32, Int32)
  JointTransformerMlp(Int32, Int32, Int32, Int32)
  VideoFinalRmsNorm(Int32, Int32, Int32)
  AudioFinalRmsNorm(Int32, Int32, Int32)
  VideoFinalAdaLnF32Boundary(Int32)
  AudioFinalAdaLnF32Boundary(Int32)
  FinalAdaLnParameterRow(Int32, Int32)
  PackedModalityAdaLnRows(Int32)
  TimestepFrequencyRows(Int32)
  TimestepEmbeddingMlpRows(Int32, Int32, Int32)
  DistinctTimestepAdaLnParameterRows(Int32, Int32)
  VideoOutputProjection(Int32, Int32, Int32, Int32, Int32)
  AudioOutputProjection(Int32, Int32)
  ShiftedFlowStep(
    @joint_diffusion_plan.JointDiffusionLatentKind,
    Int32,
    Int32,
    Int32
  )
  VideoVaeLatentDestandardization(
    Int32,
    JointDiffusionVaeLatentDestandardizationStorage,
    JointDiffusionVideoVaeLatentLayout
  )
  VideoVaeDecoderBody(Int32, Int32, Int32, Int32)
  AudioVaeLatentDestandardization(
    Int32,
    Int32,
    JointDiffusionVaeLatentDestandardizationStorage,
    JointDiffusionAudioVaeLatentLayout
  )
  AudioVaeDecoderInputProjection(
    Int32,
    Int32,
    Int32,
    Int32,
    JointDiffusionAudioVaeInputProjectionStorage,
    JointDiffusionAudioVaeLatentLayout
  )
  AudioVaeDecoderConvPre(
    Int32,
    Int32,
    Int32,
    Int32,
    Int32,
    Int32,
    JointDiffusionAudioVaeConvPreStorage,
    JointDiffusionAudioVaeWeightNormAxes,
    JointDiffusionAudioVaeLatentLayout
  )
  AudioVaeDecoderFirstUpsample(
    Int32,
    Int32,
    Int32,
    Int32,
    Int32,
    Int32,
    JointDiffusionAudioVaeConvTransposeStorage,
    JointDiffusionAudioVaeWeightNormAxes,
    JointDiffusionAudioVaeLatentLayout
  )
  AudioVaeDecoderFirstAmpActivation(
    Int32,
    Int32,
    Int32,
    Int32,
    JointDiffusionAudioVaeAliasFreeActivationStorage,
    JointDiffusionAudioVaeLatentLayout
  )
  AudioVaeDecoderFirstAmpStage(
    Int32,
    Int32,
    Int32,
    Int32,
    JointDiffusionAudioVaeFirstAmpStageStorage,
    JointDiffusionAudioVaeLatentLayout
  )
  AudioVaeDecoderUpsampleTransition(
    Int32,
    Int32,
    Int32,
    Int32,
    Int32,
    Int32,
    Int32,
    JointDiffusionAudioVaeConvTransposeStorage,
    JointDiffusionAudioVaeWeightNormAxes,
    JointDiffusionAudioVaeLatentLayout
  )
  AudioVaeDecoderAmpActivation(
    Int32,
    Int32,
    Int32,
    Int32,
    Int32,
    JointDiffusionAudioVaeAliasFreeActivationStorage,
    JointDiffusionAudioVaeLatentLayout
  )
  AudioVaeDecoderAmpStage(
    Int32,
    Int32,
    Int32,
    Int32,
    Int32,
    JointDiffusionAudioVaeAmpSchedule,
    JointDiffusionAudioVaeAmpStageStorage,
    JointDiffusionAudioVaeLatentLayout
  )
  AudioVaeDecoderBody(Int32, Int32, Int32, Int32)
} derive(Debug, Eq)

///|
/// Exact storage boundary for decode-side inverse latent normalization.
pub(all) enum JointDiffusionVaeLatentDestandardizationStorage {
  Float32InputScaleMeanOutput
} derive(Debug, Eq)

///|
/// Contiguous video-latent ordering before the VAE decoder.
pub(all) enum JointDiffusionVideoVaeLatentLayout {
  BatchChannelFrameHeightWidth
} derive(Debug, Eq)

///|
/// Contiguous audio-latent ordering before the VAE decoder.
pub(all) enum JointDiffusionAudioVaeLatentLayout {
  BatchChannelTime
} derive(Debug, Eq)

///|
/// Exact released AudioVAE decoder-input projection storage boundary.
pub(all) enum JointDiffusionAudioVaeInputProjectionStorage {
  Float32InputWeightBiasOutput
} derive(Debug, Eq)

///|
pub(all) enum JointDiffusionAudioVaeConvPreStorage {
  Float32InputWeightGWeightVBiasWorkspaceOutput
} derive(Debug, Eq)

///|
pub(all) enum JointDiffusionAudioVaeConvTransposeStorage {
  Float32InputWeightGWeightVBiasWorkspaceOutput
} derive(Debug, Eq)

///|
pub(all) enum JointDiffusionAudioVaeAliasFreeActivationStorage {
  Float32InputAlphaBetaUpsampleFilterDownsampleFilterWorkspacesOutput
} derive(Debug, Eq)

///|
pub(all) enum JointDiffusionAudioVaeFirstAmpStageStorage {
  Float32OriginalPrecomputedActivationPackedParametersWorkspaceOutput
} derive(Debug, Eq)

///|
pub(all) enum JointDiffusionAudioVaeAmpStageStorage {
  Float32OriginalPrecomputedActivationPackedParametersWorkspaceOutput
} derive(Debug, Eq)

///|
pub(all) enum JointDiffusionAudioVaeAmpSchedule {
  Kernels3_7_11Dilations1_3_5
} derive(Debug, Eq)

///|
pub(all) enum JointDiffusionAudioVaeWeightNormAxes {
  PerOutputChannelInputAndKernel
  PerInputChannelOutputAndKernel
} derive(Debug, Eq)

///|
/// Semantic capability names for future content-addressed AOT artifacts.
/// These values are requirements, not artifacts or availability evidence.
pub(all) enum JointDiffusionAotCapability {
  TokenTextConditioningAot
  OrderedKeyframeConditioningAot
  OrderedReferenceConditioningAot
  VideoLatentProjectionAot
  AudioLatentProjectionAot
  JointTransformerAttentionAot
  JointTransformerMlpAot
  VideoFinalRmsNormAot
  AudioFinalRmsNormAot
  VideoFinalAdaLnF32BoundaryAot
  AudioFinalAdaLnF32BoundaryAot
  FinalAdaLnParameterRowAot
  PackedModalityAdaLnRowsAot
  TimestepFrequencyRowsAot
  TimestepEmbeddingMlpRowsAot
  DistinctTimestepAdaLnParameterRowsAot
  VideoOutputProjectionAot
  AudioOutputProjectionAot
  ShiftedFlowStepAot
  VideoVaeLatentDestandardizationAot
  VideoVaeDecoderBodyAot
  AudioVaeLatentDestandardizationAot
  AudioVaeDecoderInputProjectionAot
  AudioVaeDecoderConvPreAot
  AudioVaeDecoderFirstUpsampleAot
  AudioVaeDecoderFirstAmpActivationAot
  AudioVaeDecoderFirstAmpStageAot
  AudioVaeDecoderUpsampleTransitionAot
  AudioVaeDecoderAmpActivationAot
  AudioVaeDecoderAmpStageAot
  AudioVaeDecoderBodyAot
} derive(Debug, Eq)

///|
/// One unique semantic operation paired with its required AOT capability.
pub struct JointDiffusionKernelRequirement {
  priv operation : JointDiffusionSemanticOperation
  priv capability : JointDiffusionAotCapability
} derive(Debug, Eq)

///|
pub fn JointDiffusionKernelRequirement::operation(
  self : Self,
) -> JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionKernelRequirement::capability(
  self : Self,
) -> JointDiffusionAotCapability {
  self.capability
}

///|
/// Digest of the source identity plus canonical ordered requirements.
pub struct JointDiffusionKernelRequirementsDigest {
  priv hex : String
} derive(Debug, Eq)

///|
pub fn JointDiffusionKernelRequirementsDigest::as_hex(self : Self) -> String {
  self.hex
}

///|
/// Immutable startup capability requirements derived from one exact plan.
pub struct JointDiffusionKernelRequirements {
  priv source_identity : @spec.ModelIdentity
  priv source_schema_version : Int32
  priv shape_contract : @joint_diffusion_plan.JointDiffusionShapeContract
  priv conditioning_limits : @joint_diffusion_plan.JointDiffusionConditioningLimits
  priv denoiser_shape : @joint_diffusion_plan.JointDiffusionDenoiserShape
  priv projection_contract : @joint_diffusion_plan.JointDiffusionProjectionContract
  priv digest : JointDiffusionKernelRequirementsDigest
  priv requirements : ReadOnlyArray[JointDiffusionKernelRequirement]
  priv capabilities : ReadOnlyArray[JointDiffusionAotCapability]
} derive(Debug, Eq)

///|
pub fn JointDiffusionKernelRequirements::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionKernelRequirements::shape_contract(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionShapeContract {
  self.shape_contract
}

///|
pub fn JointDiffusionKernelRequirements::conditioning_limits(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionConditioningLimits {
  self.conditioning_limits
}

///|
pub fn JointDiffusionKernelRequirements::denoiser_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionDenoiserShape {
  self.denoiser_shape
}

///|
pub fn JointDiffusionKernelRequirements::projection_contract(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionProjectionContract {
  self.projection_contract
}

///|
pub fn JointDiffusionKernelRequirements::source_schema_version(
  self : Self,
) -> Int32 {
  self.source_schema_version
}

///|
pub fn JointDiffusionKernelRequirements::digest(
  self : Self,
) -> JointDiffusionKernelRequirementsDigest {
  self.digest
}

///|
pub fn JointDiffusionKernelRequirements::requirements(
  self : Self,
) -> ReadOnlyArray[JointDiffusionKernelRequirement] {
  self.requirements
}

///|
pub fn JointDiffusionKernelRequirements::capabilities(
  self : Self,
) -> ReadOnlyArray[JointDiffusionAotCapability] {
  self.capabilities
}
