///|
pub(all) suberror DeepSeekV4OutputAParameterPlanError {
  InvalidEvidence
  ModelIdentityMismatch
  RequirementsMismatch
  UnsupportedOperation
  CandidateMismatch
  ConversionContractMismatch
  ShapeMismatch
  LayoutDigestMismatch
  OperandMismatch
  NoConversionAuthority
  NoMaterializationAuthority
  NoUploadAuthority
  NoLaunchAuthority
} derive(Debug, Eq)

///|
pub(all) enum DeepSeekV4OutputAParameterPlanStatus {
  InertUnqualified
} derive(Debug, Eq)

///|
pub struct DeepSeekV4OutputAParameterEvidenceInput {
  priv contract : @deepseek_v4_weights.DeepSeekV4OutputAConversionContract
  priv source_weight_shape : ReadOnlyArray[Int64]
  priv source_scale_shape : ReadOnlyArray[Int64]
  priv converted_grouped_shape : ReadOnlyArray[Int64]
  priv layout_digest_sha256 : String
  priv operand_ordinal : Int
  priv operand_role : @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperandRole
  priv operand_byte_count : Int64
  priv operand_alignment : Int64
}

///|
pub struct DeepSeekV4OutputAParameterPlan {
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv contract : @deepseek_v4_weights.DeepSeekV4OutputAConversionContract
  priv source_weight_shape : ReadOnlyArray[Int64]
  priv source_scale_shape : ReadOnlyArray[Int64]
  priv converted_grouped_shape : ReadOnlyArray[Int64]
  priv layout_digest_sha256 : String
  priv operand_ordinal : Int
  priv operand_role : @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperandRole
  priv operand_byte_count : Int64
  priv operand_alignment : Int64
  priv group_count : Int
  priv output_rank : Int
  priv input_width_per_group : Int
  priv candidate_source_digest : String
  priv candidate_recipe_digest : String
  priv digest_sha256 : String
}
