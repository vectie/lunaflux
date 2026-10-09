///|
pub(all) enum DeepSeekV4TokenHashCompileEvidenceBindingStatus {
  InertSelfConsistentUnqualified
} derive(Debug, Eq)

///|
pub struct DeepSeekV4TokenHashCompileEvidenceBindingDigest(String) derive (
  Debug,
  Eq,
  Hash,
)

///|
pub struct DeepSeekV4TokenHashCompileEvidenceBinding {
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest
  priv layer_index : Int
  priv requirement_ordinal : Int
  priv artifact_binding_digest : DeepSeekV4TokenHashArtifactBindingDigest
  priv compile_evidence_digest : @cuda_offline_compile_evidence.CudaOfflineCompileEvidenceDigest
  priv source_digest_sha256 : String
  priv recipe_digest_sha256 : String
  priv driver_identity_sha256 : String
  priv target : @catalog.DeviceTarget
  priv function_symbol : String
  priv module_digest : @catalog.AotArtifactDigest
  priv module_byte_count : Int
  priv digest : DeepSeekV4TokenHashCompileEvidenceBindingDigest
}

///|
pub(all) suberror DeepSeekV4TokenHashCompileEvidenceBindingError {
  CandidateIdentityMismatch
  CandidateRequirementsDigestMismatch
  CandidateOrdinalMismatch
  CandidateOperationMismatch
  CandidateCapabilityMismatch
  CandidateContractMismatch
  CandidateSourceDigestMismatch
  CandidateRecipeDigestMismatch
  CandidateUnexpectedlyBindable
  EvidenceUnexpectedStatus
  EvidenceSourceDigestMismatch
  EvidenceRecipeDigestMismatch
  EvidenceCompilerMismatch
  EvidenceTargetMismatch
  EvidenceSymbolMismatch
  EvidenceModuleDigestMismatch
  EvidenceModuleByteCountMismatch
  CompilationProvenanceUnavailable
  LoaderAuthorityUnavailable
  LaunchAuthorityUnavailable
  PhysicalQualificationUnavailable
} derive(Debug, Eq)
