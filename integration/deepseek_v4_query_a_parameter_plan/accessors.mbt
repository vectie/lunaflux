///|
pub fn DeepSeekV4QueryAParameterPlan::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn DeepSeekV4QueryAParameterPlan::requirement_ordinal(self : Self) -> Int {
  self.requirement_ordinal
}

///|
pub fn DeepSeekV4QueryAParameterPlan::layout_digest_sha256(
  self : Self,
) -> String {
  self.layout_digest_sha256
}

///|
pub fn DeepSeekV4QueryAParameterPlan::input_width(self : Self) -> Int {
  self.input_width
}

///|
pub fn DeepSeekV4QueryAParameterPlan::output_width(self : Self) -> Int {
  self.output_width
}

///|
pub fn DeepSeekV4QueryAParameterPlan::block_width(self : Self) -> Int {
  self.block_width
}

///|
pub fn DeepSeekV4QueryAParameterPlan::candidate_source_digest(
  self : Self,
) -> String {
  self.candidate_source_digest
}

///|
pub fn DeepSeekV4QueryAParameterPlan::candidate_recipe_digest(
  self : Self,
) -> String {
  self.candidate_recipe_digest
}

///|
pub fn DeepSeekV4QueryAParameterPlan::digest_sha256(self : Self) -> String {
  self.digest_sha256
}

///|
pub fn DeepSeekV4QueryAParameterPlan::status(
  _self : Self,
) -> DeepSeekV4QueryAParameterPlanStatus {
  InertUnqualified
}

///|
pub fn DeepSeekV4QueryAParameterPlan::require_interpretation_authority(
  _self : Self,
) -> Unit raise DeepSeekV4QueryAParameterPlanError {
  raise NoInterpretationAuthority
}

///|
pub fn DeepSeekV4QueryAParameterPlan::require_materialization_authority(
  _self : Self,
) -> Unit raise DeepSeekV4QueryAParameterPlanError {
  raise NoMaterializationAuthority
}

///|
pub fn DeepSeekV4QueryAParameterPlan::require_upload_authority(
  _self : Self,
) -> Unit raise DeepSeekV4QueryAParameterPlanError {
  raise NoUploadAuthority
}

///|
pub fn DeepSeekV4QueryAParameterPlan::require_launch_authority(
  _self : Self,
) -> Unit raise DeepSeekV4QueryAParameterPlanError {
  raise NoLaunchAuthority
}
