///|
pub(all) enum DeepSeekV4TokenHashArtifactBindingStatus {
  InertUnqualified
} derive(Debug, Eq)

///|
pub struct DeepSeekV4TokenHashArtifactBindingDigest(String) derive (
  Debug,
  Eq,
  Hash,
)

///|
pub struct DeepSeekV4TokenHashArtifactBinding {
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest
  priv layer_index : Int
  priv requirement_ordinal : Int
  priv operand_plan_digest : DeepSeekV4TokenHashOperandPlanDigest
  priv candidate_source_digest : @advanced_decoder_cuda_aot.AdvancedDecoderCudaSourceDigest
  priv candidate_recipe_digest : @advanced_decoder_cuda_aot.AdvancedDecoderCudaRecipeDigest
  priv artifact_admission_digest : @advanced_decoder_artifact_admission.AdvancedDecoderArtifactAdmissionDigest
  priv target : @catalog.DeviceTarget
  priv module_digest : @catalog.AotArtifactDigest
  priv module_byte_count : Int
  priv entry_point : @catalog.AotKernelEntryPoint
  priv function_symbol : String
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operand_count : Int
  priv token_hash_operand_ordinal : Int
  priv token_hash_device_offset : Int64
  priv digest : DeepSeekV4TokenHashArtifactBindingDigest
}

///|
pub(all) suberror DeepSeekV4TokenHashArtifactBindingError {
  CandidateIdentityMismatch
  CandidateRequirementsDigestMismatch
  CandidateOrdinalMismatch
  CandidateOperationMismatch
  CandidateCapabilityMismatch
  CandidateContractMismatch
  CandidateSourceDigestMismatch
  CandidateRecipeDigestMismatch
  CandidateUnexpectedlyBindable
  CandidateOperandMismatch(Int)
  ArtifactIdentityMismatch
  ArtifactRequirementsDigestMismatch
  ArtifactTargetMismatch
  ArtifactUnexpectedStatus
  ArtifactLaunchMissing(Int)
  ArtifactLaunchMismatch
  ArtifactOperandMismatch(Int)
  ArtifactEntryPointMissing
  ArtifactSymbolMismatch
  ArtifactModuleMissing
  ArtifactModuleMismatch
  CompilationProvenanceUnavailable
  LoaderAuthorityUnavailable
  LaunchAuthorityUnavailable
  PhysicalQualificationUnavailable
} derive(Debug, Eq)
