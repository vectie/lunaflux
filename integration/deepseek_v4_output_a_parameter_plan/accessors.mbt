///|
pub fn DeepSeekV4OutputAParameterPlan::status(
  _self : Self,
) -> DeepSeekV4OutputAParameterPlanStatus {
  InertUnqualified
}

///|
pub fn DeepSeekV4OutputAParameterPlan::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn DeepSeekV4OutputAParameterPlan::requirements_digest(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn DeepSeekV4OutputAParameterPlan::requirement_ordinal(self : Self) -> Int {
  self.requirement_ordinal
}

///|
pub fn DeepSeekV4OutputAParameterPlan::contract(
  self : Self,
) -> @deepseek_v4_weights.DeepSeekV4OutputAConversionContract {
  self.contract
}

///|
pub fn DeepSeekV4OutputAParameterPlan::source_weight_shape(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.source_weight_shape
}

///|
pub fn DeepSeekV4OutputAParameterPlan::source_scale_shape(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.source_scale_shape
}

///|
pub fn DeepSeekV4OutputAParameterPlan::converted_grouped_shape(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.converted_grouped_shape
}

///|
pub fn DeepSeekV4OutputAParameterPlan::layout_digest_sha256(
  self : Self,
) -> String {
  self.layout_digest_sha256
}

///|
pub fn DeepSeekV4OutputAParameterPlan::operand_ordinal(self : Self) -> Int {
  self.operand_ordinal
}

///|
pub fn DeepSeekV4OutputAParameterPlan::operand_role(
  self : Self,
) -> @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperandRole {
  self.operand_role
}

///|
pub fn DeepSeekV4OutputAParameterPlan::operand_byte_count(self : Self) -> Int64 {
  self.operand_byte_count
}

///|
pub fn DeepSeekV4OutputAParameterPlan::operand_alignment(self : Self) -> Int64 {
  self.operand_alignment
}

///|
pub fn DeepSeekV4OutputAParameterPlan::group_count(self : Self) -> Int {
  self.group_count
}

///|
pub fn DeepSeekV4OutputAParameterPlan::output_rank(self : Self) -> Int {
  self.output_rank
}

///|
pub fn DeepSeekV4OutputAParameterPlan::input_width_per_group(
  self : Self,
) -> Int {
  self.input_width_per_group
}

///|
pub fn DeepSeekV4OutputAParameterPlan::candidate_source_digest(
  self : Self,
) -> String {
  self.candidate_source_digest
}

///|
pub fn DeepSeekV4OutputAParameterPlan::candidate_recipe_digest(
  self : Self,
) -> String {
  self.candidate_recipe_digest
}

///|
pub fn DeepSeekV4OutputAParameterPlan::digest_sha256(self : Self) -> String {
  self.digest_sha256
}

///|
pub fn DeepSeekV4OutputAParameterPlan::require_conversion_authority(
  _self : Self,
) -> Unit raise DeepSeekV4OutputAParameterPlanError {
  raise NoConversionAuthority
}

///|
pub fn DeepSeekV4OutputAParameterPlan::require_materialization_authority(
  _self : Self,
) -> Unit raise DeepSeekV4OutputAParameterPlanError {
  raise NoMaterializationAuthority
}

///|
pub fn DeepSeekV4OutputAParameterPlan::require_upload_authority(
  _self : Self,
) -> Unit raise DeepSeekV4OutputAParameterPlanError {
  raise NoUploadAuthority
}

///|
pub fn DeepSeekV4OutputAParameterPlan::require_launch_authority(
  _self : Self,
) -> Unit raise DeepSeekV4OutputAParameterPlanError {
  raise NoLaunchAuthority
}
