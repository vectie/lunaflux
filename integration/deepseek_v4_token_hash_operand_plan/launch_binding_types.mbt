///|
pub(all) enum DeepSeekV4TokenHashLaunchBindingStatus {
  InertCompleteUnqualified
} derive(Debug, Eq)

///|
pub struct DeepSeekV4TokenHashLaunchBindingDigest(String) derive(Debug, Eq, Hash)

///|
pub struct DeepSeekV4TokenHashLaunchBinding {
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest
  priv layer_index : Int
  priv requirement_ordinal : Int
  priv maximum_tokens : Int
  priv experts_per_token : Int
  priv routed_expert_count : Int
  priv operand_plan_digest : DeepSeekV4TokenHashOperandPlanDigest
  priv sidecar_layout_digest : @segmented_device_materialize.SegmentedDeviceLayoutDigest
  priv artifact_binding_digest : DeepSeekV4TokenHashArtifactBindingDigest
  priv compile_binding_digest : DeepSeekV4TokenHashCompileEvidenceBindingDigest
  priv compile_evidence_digest : @cuda_offline_compile_evidence.CudaOfflineCompileEvidenceDigest
  priv operand_regions : @advanced_decoder_operand_region_plan.AdvancedDecoderOperandRegionPlan
  priv source_digest_sha256 : String
  priv recipe_digest_sha256 : String
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv target : @catalog.DeviceTarget
  priv entry_point : @catalog.AotKernelEntryPoint
  priv function_symbol : String
  priv module_digest : @catalog.AotArtifactDigest
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv digest : DeepSeekV4TokenHashLaunchBindingDigest
}

///|
pub(all) suberror DeepSeekV4TokenHashLaunchBindingError {
  ArtifactBinding(DeepSeekV4TokenHashArtifactBindingError)
  OperandRegions(
    @advanced_decoder_operand_region_plan.AdvancedDecoderOperandRegionPlanError
  )
  OperandPlanMismatch
  SidecarLayoutMismatch
  ArtifactEvidenceMismatch
  CompileEvidenceMismatch
  CandidateGeometryMismatch(Int)
  LoaderAuthorityUnavailable
  LaunchAuthorityUnavailable
  PhysicalQualificationUnavailable
} derive(Debug, Eq)
