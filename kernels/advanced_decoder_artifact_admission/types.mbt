///|
pub(all) suberror AdvancedDecoderArtifactAdmissionError {
  InvalidLimit
  InvalidDeclaredQualificationDigest
  RequirementCountExceeded
  RequirementDigestMismatch(Int)
  SourceIdentityMismatch(Int)
  TargetMismatch(Int)
  MissingLaunch(Int)
  DuplicateLaunch(Int)
  WrongOperation(Int)
  WrongCapability(Int)
  UnsupportedOperation(Int)
  EntryPointModuleMismatch(Int)
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
pub struct AdvancedDecoderArtifactAdmissionLimits {
  priv maximum_requirements : Int
  priv maximum_modules : Int
  priv maximum_entry_points : Int
  priv maximum_module_bytes : Int64
  priv maximum_total_module_bytes : Int64
  priv maximum_symbol_bytes : Int
  priv maximum_operands_per_launch : Int
  priv maximum_canonical_bytes : Int
} derive(Debug, Eq)

///|
/// Caller-declared qualification metadata is inert evidence, not authority.
pub struct AdvancedDecoderDeclaredQualificationDigest {
  priv hex : String
} derive(Debug, Eq, Hash)

///|
pub struct AdvancedDecoderModuleInput {
  priv digest : @catalog.AotArtifactDigest
  priv module_bytes : Bytes
}

///|
/// Ordered semantic ABI roles for advanced-decoder launch evidence.
pub(all) enum AdvancedDecoderLaunchOperandRole {
  StepCountsInput
  TokenIdsInput
  PositionInput
  HiddenInput
  HiddenOutput
  ModelWeights
  HyperConnectionStateInput
  HyperConnectionStateOutput
  HyperConnectionFunctionWeightsInput
  HyperConnectionBaseInput
  HyperConnectionScaleInput
  HyperConnectionPreOutput
  HyperConnectionPostOutput
  HyperConnectionCombinationOutput
  HyperConnectionPreInput
  HyperConnectionReducedOutput
  HyperConnectionBranchInput
  HyperConnectionPostInput
  HyperConnectionCombinationInput
  HyperConnectionCombinedOutput
  RoutingScoresInput
  RoutingScoresOutput
  RoutingIndicesInput
  RoutingIndicesOutput
  CompressedQueryPositionsInput
  CompressedCacheOffsetsInput
  CompressedIndexCountsOutput
  CompressedIndicesOutput
  SparseAttentionQueryInput
  SparseAttentionQueryBatchIdsInput
  SparseAttentionKeyValueInput
  SparseAttentionCacheLengthsInput
  SparseAttentionSinkInput
  SparseAttentionSelectedIndicesInput
  SparseAttentionOutput
  WindowQueryPositionsInput
  WindowSequenceLengthsInput
  WindowQueryModesInput
  CompressedIndexCountsInput
  CompressedIndicesInput
  JoinedIndexCountsOutput
  JoinedIndicesOutput
  PreparedKeyValueModesInput
  PreparedKeyValueSequenceLengthsInput
  UncompressedKeyValueInput
  CompressedKeyValueInput
  PreparedKeyValueLengthsOutput
  PreparedKeyValueOutput
  AttentionOutputRotaryInOut
  AttentionOutputProjectionInput
  AttentionOutputAWeightsInput
  AttentionOutputRankOutput
  AttentionOutputRankInput
  AttentionOutputBWeightCodesInput
  AttentionOutputBWeightScalesInput
  QueryAWeightCodesInput
  QueryAWeightScalesInput
  QueryLowRankAOutput
  KeyValueWeightCodesInput
  KeyValueWeightScalesInput
  KeyValueProjectionOutput
  QueryInput
  QueryNormWeights
  QueryOutput
  KeyValueInput
  KeyValueNormWeights
  KeyValueOutput
  TokenHashTableInput
  RoutingSelectionBiasInput
  RoutingWeightsOutput
  ExpertInput
  ExpertOutput
  AuxiliaryInput
  AuxiliaryOutput
  LogitsOutput
  Workspace
} derive(Debug, Eq)

///|
pub struct AdvancedDecoderLaunchOperand {
  priv role : AdvancedDecoderLaunchOperandRole
  priv byte_count : Int64
  priv alignment : Int64
} derive(Debug, Eq)

///|
/// Untrusted evidence for one required semantic operation.
pub struct AdvancedDecoderLaunchEvidenceInput {
  priv requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest
  priv source_identity : @spec.ModelIdentity
  priv requirement_ordinal : Int
  priv operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation
  priv capability : @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability
  priv target : @catalog.DeviceTarget
  priv entry_point : @catalog.AotKernelEntryPoint
  priv module_digest : @catalog.AotArtifactDigest
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[AdvancedDecoderLaunchOperand]
  priv declared_qualification : AdvancedDecoderDeclaredQualificationDigest?
} derive(Debug, Eq)

///|
pub struct AdvancedDecoderAdmittedModule {
  priv digest : @catalog.AotArtifactDigest
  priv module_bytes : Bytes
}

///|
pub struct AdvancedDecoderAdmittedEntryPoint {
  priv identity : @catalog.AotKernelEntryPoint
  priv module_index : Int
  priv function_symbol : String
} derive(Debug, Eq)

///|
pub struct AdvancedDecoderAdmittedLaunch {
  priv requirement_ordinal : Int
  priv operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation
  priv capability : @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability
  priv entry_point : @catalog.AotKernelEntryPoint
  priv module_digest : @catalog.AotArtifactDigest
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[AdvancedDecoderLaunchOperand]
  priv declared_qualification : AdvancedDecoderDeclaredQualificationDigest?
} derive(Debug, Eq)

///|
pub struct AdvancedDecoderArtifactAdmissionDigest {
  priv hex : String
} derive(Debug, Eq, Hash)

///|
/// Artifact evidence is never execution authority.
pub(all) enum AdvancedDecoderArtifactAdmissionStatus {
  InertUnqualified
} derive(Debug, Eq)

///|
/// Immutable, content-addressed, non-runnable artifact evidence.
pub struct AdvancedDecoderArtifactAdmission {
  priv schema_version : Int
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest
  priv target : @catalog.DeviceTarget
  priv status : AdvancedDecoderArtifactAdmissionStatus
  priv digest : AdvancedDecoderArtifactAdmissionDigest
  priv modules : ReadOnlyArray[AdvancedDecoderAdmittedModule]
  priv entry_points : ReadOnlyArray[AdvancedDecoderAdmittedEntryPoint]
  priv launches : ReadOnlyArray[AdvancedDecoderAdmittedLaunch]
  priv total_module_bytes : Int64
}
