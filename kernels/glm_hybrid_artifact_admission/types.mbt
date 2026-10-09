///|
pub(all) suberror GlmHybridArtifactAdmissionError {
  InvalidLimit
  InvalidDeclaredQualificationDigest
  ClaimCountMismatch(expected~ : Int, actual~ : Int)
  MissingClaim(ordinal~ : Int)
  DuplicateClaim(ordinal~ : Int)
  DuplicateEntryPoint(ordinal~ : Int)
  SourceDigestMismatch(ordinal~ : Int)
  SourceIdentityMismatch(ordinal~ : Int)
  TargetMismatch(ordinal~ : Int)
  CapabilityMismatch(ordinal~ : Int)
  GeometryMismatch(ordinal~ : Int)
  ArtifactDigestMismatch(ordinal~ : Int)
  EntryPointModuleMismatch(ordinal~ : Int)
  InvalidLaunch(ordinal~ : Int)
  InvalidFunctionSymbol(ordinal~ : Int)
  InvalidLaunchOperands(ordinal~ : Int)
  AdmissionEncodingTooLarge
  NoExecutionAuthority
} derive(Debug, Eq)

///|
pub struct GlmHybridArtifactAdmissionLimits {
  priv maximum_claims : Int
  priv maximum_module_bytes : Int
  priv maximum_total_module_bytes : Int64
  priv maximum_canonical_bytes : Int
}

///|
/// Ordered semantic roles for GLM launch evidence. The current executable
/// subset uses decoder-foundation, hidden-projection, and terminal-logits roles.
pub(all) enum GlmHybridLaunchOperandRole {
  StepCountsInput
  TokenIdsInput
  HiddenInput
  ModelWeights
  HiddenOutput
  LogitsOutput
  DsaIndexKeysInput
  DsaIndexGateScoresInput
  DsaIndexValidityInput
  DsaPooledKeysOutput
  DsaPooledRawIndicesOutput
  DsaPooledValidityOutput
  DsaPooledCountOutput
  QueryPositionsInput
  DsaIndexQueriesInput
  DsaIndexHeadWeightsInput
  DsaCandidateKeysInput
  DsaCandidateRawIndicesInput
  DsaCandidateValidityInput
  DsaCandidateCountInput
  DsaKeyValidityInput
  DsaSelectedIndicesInput
  DsaSelectedIndicesOutput
  DsaKeyCountInput
  DsaAttentionQueriesInput
  DsaAttentionKeysInput
  DsaAttentionValuesInput
  DsaAttentionOutput
  DsaQueryCompressedOutput
  DsaKeyValueCompressedRopeOutput
  DsaQueryCompressedInput
  DsaQueryNormalizedOutput
  DsaQueryHeadsOutput
  DsaQueryNormalizedInput
  DsaKeyValueNormalizedInput
  DsaKeyValueHeadsOutput
  DsaKeyValueCompressedRopeInput
  DsaKeyValueNormalizedOutput
  DsaKeyRopeSuffixOutput
  DsaAttentionKeyRopeInput
  DenseGateWeights
  DenseUpWeights
  DenseDownWeights
  KdaSequenceOffsetsInput
  KdaProjectedInput
  KdaShortConvolutionWeights
  KdaConvolutionInitialStateInput
  KdaConvolvedOutput
  KdaConvolutionFinalStateOutput
  KdaRecurrentQueryInput
  KdaRecurrentKeyInput
  KdaRecurrentValueInput
  KdaLogDecayInput
  KdaInputGateInput
  KdaRecurrentInitialStateInput
  KdaRecurrentOutput
  KdaRecurrentFinalStateOutput
  KdaQueryProjectionWeights
  KdaKeyProjectionWeights
  KdaValueProjectionWeights
  KdaForgetLowRankWeights
  KdaForgetExpansionWeights
  KdaInputGateWeights
  KdaProjectedQueryOutput
  KdaProjectedKeyOutput
  KdaProjectedValueOutput
  KdaForgetLowRankOutput
  KdaRawForgetOutput
  KdaBetaOutput
  KdaOutputGateLowRankWeights
  KdaOutputGateExpansionWeights
  KdaOutputGateLowRankOutput
  KdaOutputGateOutput
  KdaGatedNormRecurrentInput
  KdaGatedNormGateInput
  KdaGatedNormWeights
  KdaGatedNormOutput
  KdaOutputProjectionInput
  KdaOutputProjectionWeights
  KdaOutputProjectionOutput
  KdaDecayRawForgetInput
  KdaDecayBiasInput
  KdaDecayLogRateInput
  KdaDecayOutput
  MoeRouterFloat32Input
  MoeRouterBf16Weights
  MoeRouterFloat32LogitsOutput
  MoeCorrectionBiasInput
  MoeUncorrectedScoresOutput
  MoeChoiceScoresOutput
  MoeSelectedIndicesOutput
  MoeSelectedWeightsOutput
  MoeFeedForwardInput
  MoeExpertGateWeights
  MoeExpertUpWeights
  MoeExpertDownWeights
  MoeRoutedOutput
  MoeSharedGateWeights
  MoeSharedUpWeights
  MoeSharedDownWeights
  MoeSharedOutput
  MoeCombinedOutput
  DecoderResidualInput
  DecoderResidualOutput
  HyperPostControlInput
  HyperCombineControlInput
  HyperResidualStreamsInput
  HyperSublayerOutputInput
  HyperMixedStreamsOutput
  HyperFunctionWeightsInput
  HyperBaseInput
  HyperScaleInput
  HyperInitialCombinationOutput
  HyperCollapsedOutput
} derive(Debug, Eq)

///|
/// One exact launch operand. Construction validates only bounded scalar
/// metadata; operation-specific ordering and sizes are checked by admission.
pub struct GlmHybridLaunchOperand {
  priv role : GlmHybridLaunchOperandRole
  priv byte_count : Int64
  priv alignment : Int64
} derive(Debug, Eq)

///|
pub fn GlmHybridLaunchOperand::new(
  role~ : GlmHybridLaunchOperandRole,
  byte_count~ : Int64,
  alignment~ : Int64,
) -> Self raise GlmHybridArtifactAdmissionError {
  if byte_count <= 0L ||
    alignment <= 0L ||
    alignment > 4096L ||
    (alignment & (alignment - 1L)) != 0L {
    raise InvalidLaunchOperands(ordinal=-1)
  }
  { role, byte_count, alignment, }
}

///|
pub fn GlmHybridArtifactAdmissionLimits::new(
  maximum_claims~ : Int,
  maximum_module_bytes~ : Int,
  maximum_total_module_bytes~ : Int64,
  maximum_canonical_bytes~ : Int,
) -> Self raise GlmHybridArtifactAdmissionError {
  if maximum_claims <= 0 ||
    maximum_claims > 4096 ||
    maximum_module_bytes <= 0 ||
    maximum_module_bytes > 1024 * 1024 * 1024 ||
    maximum_total_module_bytes <= 0L ||
    maximum_canonical_bytes <= 0 ||
    maximum_canonical_bytes > 16 * 1024 * 1024 {
    raise InvalidLimit
  }
  {
    maximum_claims,
    maximum_module_bytes,
    maximum_total_module_bytes,
    maximum_canonical_bytes,
  }
}

///|
pub struct GlmHybridDeclaredQualificationDigest {
  priv hex : String
} derive(Debug, Eq, Hash)

///|
pub fn GlmHybridDeclaredQualificationDigest::from_sha256(
  hex : String,
) -> Self raise GlmHybridArtifactAdmissionError {
  if hex.length() != 64 {
    raise InvalidDeclaredQualificationDigest
  }
  for c in hex {
    if !(c >= '0' && c <= '9') && !(c >= 'a' && c <= 'f') {
      raise InvalidDeclaredQualificationDigest
    }
  }
  { hex, }
}

///|
pub fn GlmHybridDeclaredQualificationDigest::as_hex(self : Self) -> String {
  self.hex
}

///|
pub(all) enum GlmHybridArtifactAdmissionStatus {
  NonRunnableUnqualified
  NonRunnableWithDeclaredQualificationDigests
} derive(Debug, Eq)

///|
pub struct GlmHybridArtifactClaimInput {
  priv ordinal : Int
  priv source_digest : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest
  priv source_identity : @spec.ModelIdentity
  priv target : @catalog.DeviceTarget
  priv operation : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation
  priv capability : @glm_hybrid_kernel_requirements.GlmHybridAotCapability
  priv module_digest : @catalog.AotArtifactDigest
  priv module_bytes : Bytes
  priv entry_point : @catalog.AotKernelEntryPoint
  priv function_symbol : String
  priv launch : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[GlmHybridLaunchOperand]
  priv declared_qualification : GlmHybridDeclaredQualificationDigest?
}

///|
pub fn GlmHybridArtifactClaimInput::new(
  ordinal~ : Int,
  source_digest~ : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest,
  source_identity~ : @spec.ModelIdentity,
  target~ : @catalog.DeviceTarget,
  operation~ : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation,
  capability~ : @glm_hybrid_kernel_requirements.GlmHybridAotCapability,
  module_digest~ : @catalog.AotArtifactDigest,
  module_bytes~ : Bytes,
  entry_point~ : @catalog.AotKernelEntryPoint,
  function_symbol~ : String,
  launch~ : @launch_contract.AotLaunchDimensions,
  operands~ : ArrayView[GlmHybridLaunchOperand],
  declared_qualification? : GlmHybridDeclaredQualificationDigest,
) -> Self {
  {
    ordinal,
    source_digest,
    source_identity,
    target,
    operation,
    capability,
    module_digest,
    module_bytes,
    entry_point,
    function_symbol,
    launch,
    operands: ReadOnlyArray::from_array(operands),
    declared_qualification,
  }
}

///|
pub struct GlmHybridAdmittedArtifact {
  priv ordinal : Int
  priv operation : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation
  priv capability : @glm_hybrid_kernel_requirements.GlmHybridAotCapability
  priv module_digest : @catalog.AotArtifactDigest
  priv module_byte_length : Int
  priv entry_point : @catalog.AotKernelEntryPoint
  priv function_symbol : String
  priv launch : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[GlmHybridLaunchOperand]
  priv declared_qualification : GlmHybridDeclaredQualificationDigest?
} derive(Debug, Eq)

///|
pub struct GlmHybridArtifactAdmissionDigest {
  priv hex : String
} derive(Debug, Eq, Hash)

///|
pub struct GlmHybridArtifactAdmission {
  priv source_digest : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest
  priv source_identity : @spec.ModelIdentity
  priv target : @catalog.DeviceTarget
  priv status : GlmHybridArtifactAdmissionStatus
  priv artifacts : ReadOnlyArray[GlmHybridAdmittedArtifact]
  priv digest : GlmHybridArtifactAdmissionDigest
} derive(Debug, Eq)
