///|
/// Bounded inert-admission failures. No module bytes or CUDA symbols are
/// copied into error payloads.
pub(all) suberror JointDiffusionArtifactAdmissionError {
  InvalidLimit
  RequirementCountExceeded
  RequirementDigestMismatch(Int)
  TargetMismatch(Int)
  MissingLaunch(Int)
  DuplicateLaunch(Int)
  WrongOperation(Int)
  WrongCapability(Int)
  InvalidLaunchOperands(Int)
  MissingEntryPoint(@catalog.AotKernelEntryPoint)
  DuplicateEntryPoint(@catalog.AotKernelEntryPoint)
  UnreferencedEntryPoint(@catalog.AotKernelEntryPoint)
  ArtifactSymbolFailure(@artifact.KernelArtifactError)
  MissingModule(@catalog.AotArtifactDigest)
  DuplicateModule(@catalog.AotArtifactDigest)
  UnreferencedModule(@catalog.AotArtifactDigest)
  EmptyModule(@catalog.AotArtifactDigest)
  ModuleTooLarge(@catalog.AotArtifactDigest)
  TotalModuleBytesExceeded
  ModuleDigestMismatch(@catalog.AotArtifactDigest)
  AdmissionEncodingTooLarge
  NoExecutionAuthority
} derive(Debug, Eq)

///|
/// Absolute startup ceilings for the separate diffusion evidence vocabulary.
pub struct JointDiffusionArtifactAdmissionLimits {
  priv maximum_requirements : Int
  priv maximum_modules : Int
  priv maximum_entry_points : Int
  priv maximum_module_bytes : Int64
  priv maximum_total_module_bytes : Int64
  priv maximum_symbol_bytes : Int
  priv maximum_operands_per_launch : Int
} derive(Debug, Eq)

///|
/// Immutable module snapshot. Existing artifact inputs keep their bytes
/// private behind decoder launch-contract admission, so diffusion admission
/// owns this narrow content-addressed input type.
pub struct JointDiffusionModuleInput {
  priv digest : @catalog.AotArtifactDigest
  priv module_bytes : Bytes
}

///|
/// Semantic launch operand vocabulary for joint diffusion AOT entry points.
pub(all) enum JointDiffusionLaunchOperandRole {
  TokenIdsInput
  TextConditioningOutput
  KeyframeMediaInput
  OrderedReferenceMediaInput
  OrderedConditioningOutput
  VideoLatentInput
  AudioLatentInput
  VideoPredictionInput
  AudioPredictionInput
  FlowCoefficientsInput
  JointHiddenInput
  JointHiddenOutput
  ModelWeights
  VideoLatentOutput
  AudioLatentOutput
  VideoMediaOutput
  AudioMediaOutput
  Workspace
  ProjectionWeight
  ProjectionBias
  VideoHiddenInput
  AudioHiddenInput
  VideoPredictionOutput
  AudioPredictionOutput
  VideoFinalNormInput
  AudioFinalNormInput
  FinalNormWeight
  VideoFinalNormOutput
  AudioFinalNormOutput
  FinalAdaLnShiftInput
  FinalAdaLnScaleInput
  VideoF32HeadInput
  AudioF32HeadInput
  FinalAdaLnConditioningInput
  FinalAdaLnWeight
  FinalAdaLnBias
  FinalAdaLnShiftOutput
  FinalAdaLnScaleOutput
  PackedAdaLnShiftInput
  PackedAdaLnScaleInput
  VideoPackedRowIndices
  AudioPackedRowIndices
  VideoAdaLnShiftOutput
  VideoAdaLnScaleOutput
  AudioAdaLnShiftOutput
  AudioAdaLnScaleOutput
  DistinctTimestepValuesInput
  TimestepFrequencyRowsOutput
  TimestepMlpWeight1
  TimestepMlpBias1
  TimestepMlpWeight2
  TimestepMlpBias2
  TimestepEmbeddingRowsOutput
  TransformerMlpPackedGateUpWeight
  TransformerMlpDownWeight
  AttentionQueryWeight
  AttentionKeyWeight
  AttentionValueWeight
  AttentionOutputWeight
  AttentionQueryNormWeight
  AttentionKeyNormWeight
  AttentionRotaryCosInput
  AttentionRotarySinInput
  AttentionQueryBuffer
  AttentionKeyBuffer
  AttentionValueBuffer
  AttentionNormalizedQueryBuffer
  AttentionNormalizedKeyBuffer
  AttentionOutputBuffer
  VideoNormalizedLatentInput
  VideoLatentStandardDeviationInput
  VideoLatentMeanInput
  VideoDestandardizedLatentOutput
  AudioNormalizedLatentInput
  AudioLatentStandardDeviationInput
  AudioLatentMeanInput
  AudioDestandardizedLatentOutput
  AudioDecoderInputProjectionWeight
  AudioDecoderInputProjectionBias
  AudioDecoderInputProjectionOutput
  AudioDecoderConvPreWeightG
  AudioDecoderConvPreWeightV
  AudioDecoderConvPreBias
  AudioDecoderConvPreNormalizedWeightWorkspace
  AudioDecoderConvPreOutput
  AudioDecoderFirstUpsampleWeightG
  AudioDecoderFirstUpsampleWeightV
  AudioDecoderFirstUpsampleBias
  AudioDecoderFirstUpsampleNormalizedWeightWorkspace
  AudioDecoderFirstUpsampleOutput
  AudioDecoderFirstAmpAlpha
  AudioDecoderFirstAmpBeta
  AudioDecoderFirstAmpUpsampleFilter
  AudioDecoderFirstAmpDownsampleFilter
  AudioDecoderFirstAmpUpsampleWorkspace
  AudioDecoderFirstAmpActivatedWorkspace
  AudioDecoderFirstAmpActivationOutput
  AudioDecoderFirstAmpStageOutput
  AudioDecoderUpsampleTransitionWeightG
  AudioDecoderUpsampleTransitionWeightV
  AudioDecoderUpsampleTransitionBias
  AudioDecoderUpsampleTransitionNormalizedWeightWorkspace
  AudioDecoderUpsampleTransitionOutput
  AudioDecoderAmpAlpha
  AudioDecoderAmpBeta
  AudioDecoderAmpUpsampleFilter
  AudioDecoderAmpDownsampleFilter
  AudioDecoderAmpUpsampleWorkspace
  AudioDecoderAmpActivatedWorkspace
  AudioDecoderAmpActivationOutput
  AudioDecoderAmpStageOutput
  AudioDecoderUpsampleTransition2WeightG
  AudioDecoderUpsampleTransition2WeightV
  AudioDecoderUpsampleTransition2Bias
  AudioDecoderUpsampleTransition2NormalizedWeightWorkspace
  AudioDecoderUpsampleTransition2Output
  AudioDecoderAmpActivation2Output
} derive(Debug, Eq)

///|
/// One exact ordered semantic operand with inert byte/alignment evidence.
pub struct JointDiffusionLaunchOperand {
  priv role : JointDiffusionLaunchOperandRole
  priv byte_count : Int64
  priv alignment : Int64
} derive(Debug, Eq)

///|
/// Untrusted evidence for one required semantic launch.
pub struct JointDiffusionLaunchEvidenceInput {
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation
  priv capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability
  priv target : @catalog.DeviceTarget
  priv entry_point : @catalog.AotKernelEntryPoint
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[JointDiffusionLaunchOperand]
} derive(Debug, Eq)

///|
/// One admitted immutable module, retained once by content digest.
pub struct JointDiffusionAdmittedModule {
  priv digest : @catalog.AotArtifactDigest
  priv module_bytes : Bytes
}

///|
/// One admitted stable entry-point identity and bounded CUDA symbol mapping.
pub struct JointDiffusionAdmittedEntryPoint {
  priv identity : @catalog.AotKernelEntryPoint
  priv module_index : Int
  priv function_symbol : String
} derive(Debug, Eq)

///|
/// Exact admitted launch evidence in requirement order.
pub struct JointDiffusionAdmittedLaunch {
  priv requirement_ordinal : Int
  priv operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation
  priv capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability
  priv entry_point : @catalog.AotKernelEntryPoint
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[JointDiffusionLaunchOperand]
} derive(Debug, Eq)

///|
/// Digest of requirements plus target, module, symbol, and launch evidence.
pub struct JointDiffusionArtifactAdmissionDigest {
  priv hex : String
} derive(Debug, Eq)

///|
/// This admission layer authenticates inert metadata only.
pub(all) enum JointDiffusionArtifactAdmissionStatus {
  InertUnqualified
} derive(Debug, Eq)

///|
/// Immutable inert admission result. It grants no loading or execution
/// authority and contains no device handle.
pub struct JointDiffusionArtifactAdmission {
  priv schema_version : Int
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv target : @catalog.DeviceTarget
  priv status : JointDiffusionArtifactAdmissionStatus
  priv digest : JointDiffusionArtifactAdmissionDigest
  priv modules : ReadOnlyArray[JointDiffusionAdmittedModule]
  priv entry_points : ReadOnlyArray[JointDiffusionAdmittedEntryPoint]
  priv launches : ReadOnlyArray[JointDiffusionAdmittedLaunch]
  priv total_module_bytes : Int64
}
