///|
pub fn DeepSeekV4OutputBParameterPlan::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn DeepSeekV4OutputBParameterPlan::requirement_ordinal(self : Self) -> Int {
  self.requirement_ordinal
}

///|
pub fn DeepSeekV4OutputBParameterPlan::layout_digest_sha256(
  self : Self,
) -> String {
  self.layout_digest_sha256
}

///|
pub fn DeepSeekV4OutputBParameterPlan::weight_shape(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.weight_shape
}

///|
pub fn DeepSeekV4OutputBParameterPlan::scale_shape(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.scale_shape
}

///|
pub fn DeepSeekV4OutputBParameterPlan::input_width(self : Self) -> Int {
  self.input_width
}

///|
pub fn DeepSeekV4OutputBParameterPlan::output_width(self : Self) -> Int {
  self.output_width
}

///|
pub fn DeepSeekV4OutputBParameterPlan::block_width(self : Self) -> Int {
  self.block_width
}

///|
pub fn DeepSeekV4OutputBParameterPlan::candidate_source_digest(
  self : Self,
) -> String {
  self.candidate_source_digest
}

///|
pub fn DeepSeekV4OutputBParameterPlan::candidate_recipe_digest(
  self : Self,
) -> String {
  self.candidate_recipe_digest
}

///|
pub fn DeepSeekV4OutputBParameterPlan::digest_sha256(self : Self) -> String {
  self.digest_sha256
}

///|
pub fn DeepSeekV4OutputBParameterPlan::status(
  _self : Self,
) -> DeepSeekV4OutputBParameterPlanStatus {
  InertUnqualified
}

///|
pub fn DeepSeekV4OutputBParameterPlan::require_interpretation_authority(
  _self : Self,
) -> Unit raise DeepSeekV4OutputBParameterPlanError {
  raise NoInterpretationAuthority
}

///|
pub fn DeepSeekV4OutputBParameterPlan::require_materialization_authority(
  _self : Self,
) -> Unit raise DeepSeekV4OutputBParameterPlanError {
  raise NoMaterializationAuthority
}

///|
pub fn DeepSeekV4OutputBParameterPlan::require_upload_authority(
  _self : Self,
) -> Unit raise DeepSeekV4OutputBParameterPlanError {
  raise NoUploadAuthority
}

///|
pub fn DeepSeekV4OutputBParameterPlan::require_collective_authority(
  _self : Self,
) -> Unit raise DeepSeekV4OutputBParameterPlanError {
  raise NoCollectiveAuthority
}

///|
pub fn DeepSeekV4OutputBParameterPlan::require_launch_authority(
  _self : Self,
) -> Unit raise DeepSeekV4OutputBParameterPlanError {
  raise NoLaunchAuthority
}
