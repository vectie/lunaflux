///|
pub(all) suberror DeepSeekV4OutputBParameterPlanError {
  InvalidEvidence
  ModelIdentityMismatch
  RequirementsMismatch
  UnsupportedOperation
  CandidateMismatch
  QuantizationContractMismatch
  ShapeMismatch
  LayoutDigestMismatch
  OperandMismatch
  NoInterpretationAuthority
  NoMaterializationAuthority
  NoUploadAuthority
  NoCollectiveAuthority
  NoLaunchAuthority
} derive(Debug, Eq)

///|
pub(all) enum DeepSeekV4OutputBParameterPlanStatus {
  InertUnqualified
} derive(Debug, Eq)

///|
pub struct DeepSeekV4OutputBParameterEvidenceInput {
  priv contract : @deepseek_v4_weights.DeepSeekV4OutputBQuantizationContract
  priv weight_shape : ReadOnlyArray[Int64]
  priv scale_shape : ReadOnlyArray[Int64]
  priv layout_digest_sha256 : String
  priv weight_operand_ordinal : Int
  priv weight_operand_role : @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperandRole
  priv weight_operand_bytes : Int64
  priv scale_operand_ordinal : Int
  priv scale_operand_role : @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperandRole
  priv scale_operand_bytes : Int64
}

///|
pub struct DeepSeekV4OutputBParameterPlan {
  priv source_identity : @spec.ModelIdentity
  priv requirement_ordinal : Int
  priv layout_digest_sha256 : String
  priv weight_shape : ReadOnlyArray[Int64]
  priv scale_shape : ReadOnlyArray[Int64]
  priv input_width : Int
  priv output_width : Int
  priv block_width : Int
  priv candidate_source_digest : String
  priv candidate_recipe_digest : String
  priv digest_sha256 : String
}
