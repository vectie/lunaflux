///|
pub(all) enum DeepSeekV4TokenHashOperandPlanStatus {
  InertUnqualified
} derive(Debug, Eq)

///|
pub struct DeepSeekV4TokenHashOperandPlanDigest(String) derive(Debug, Eq, Hash)

///|
pub struct DeepSeekV4TokenHashOperandPlan {
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest
  priv candidate_source_digest : @advanced_decoder_cuda_aot.AdvancedDecoderCudaSourceDigest
  priv candidate_recipe_digest : @advanced_decoder_cuda_aot.AdvancedDecoderCudaRecipeDigest
  priv sidecar_layout_digest : @segmented_device_materialize.SegmentedDeviceLayoutDigest
  priv sidecar_source_digest : @segmented_device_materialize.SegmentedHostSourceDigest
  priv source_manifest_digest : String
  priv table_source_label : String
  priv requirement_ordinal : Int32
  priv layer_index : Int32
  priv routed_expert_count : Int32
  priv operand_ordinal : Int32
  priv operand_role : @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperandRole
  priv byte_count : Int64
  priv alignment : Int64
  priv region_ordinal : Int32
  priv region_tensor_ordinal : Int32
  priv device_offset : Int64
  priv digest : DeepSeekV4TokenHashOperandPlanDigest
}

///|
pub(all) suberror DeepSeekV4TokenHashOperandPlanError {
  Sidecar(@deepseek_v4_device_upload.DeepSeekV4DeviceUploadError)
  TokenHash(
    @deepseek_v4_host_materialization.DeepSeekV4TokenHashMaterializationError
  )
  SourceReleased
  InvalidSourceMetadata
  InvalidLayer(Int32)
  SourceIdentityMismatch
  RequirementsIdentityMismatch
  RequirementsDigestMismatch
  CandidateRequirementOrdinalMismatch(Int32)
  CandidateOperationMismatch
  CandidateCapabilityMismatch
  CandidateContractMismatch
  CandidateUnexpectedlyBindable
  OperandOrdinalMismatch
  OperandRoleMismatch
  OperandByteCountMismatch
  OperandAlignmentMismatch
  LayoutRegionMismatch(Int32)
  NoLaunchAuthority
} derive(Debug, Eq)

///|
priv struct TokenHashTableEvidence {
  layer_index : Int32
  digest : String
  byte_count : Int32
}

///|
priv struct TokenHashSourceEvidence {
  identity : @spec.ModelIdentity
  source_manifest_digest : String
  tables : ReadOnlyArray[TokenHashTableEvidence]
  released : Bool
}

///|
priv struct TokenHashCandidateEvidence {
  source_identity : @spec.ModelIdentity
  requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest
  requirement_ordinal : Int32
  operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation
  capability : @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability
  family : @advanced_decoder_cuda_aot.AdvancedDecoderRoutingAotFamily
  numerical_contract : @advanced_decoder_cuda_aot.AdvancedDecoderRoutingNumericalContract
  routed_expert_count : Int32
  operand_count : Int32
  operand_role : @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperandRole
  operand_byte_count : Int64
  operand_alignment : Int64
  source_digest : @advanced_decoder_cuda_aot.AdvancedDecoderCudaSourceDigest
  recipe_digest : @advanced_decoder_cuda_aot.AdvancedDecoderCudaRecipeDigest
  manifest_bindable : Bool
}
