///|
pub fn DeepSeekV4KeyValueParameterPlan::status(
  _self : Self,
) -> DeepSeekV4KeyValueParameterPlanStatus {
  InertUnqualified
}

///|
pub fn DeepSeekV4KeyValueParameterPlan::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn DeepSeekV4KeyValueParameterPlan::requirement_ordinal(self : Self) -> Int {
  self.requirement_ordinal
}

///|
pub fn DeepSeekV4KeyValueParameterPlan::layout_digest_sha256(
  self : Self,
) -> String {
  self.layout_digest_sha256
}

///|
pub fn DeepSeekV4KeyValueParameterPlan::input_width(self : Self) -> Int {
  self.input_width
}

///|
pub fn DeepSeekV4KeyValueParameterPlan::output_width(self : Self) -> Int {
  self.output_width
}

///|
pub fn DeepSeekV4KeyValueParameterPlan::rope_width(self : Self) -> Int {
  self.rope_width
}

///|
pub fn DeepSeekV4KeyValueParameterPlan::block_width(self : Self) -> Int {
  self.block_width
}

///|
pub fn DeepSeekV4KeyValueParameterPlan::candidate_source_digest(
  self : Self,
) -> String {
  self.candidate_source_digest
}

///|
pub fn DeepSeekV4KeyValueParameterPlan::candidate_recipe_digest(
  self : Self,
) -> String {
  self.candidate_recipe_digest
}

///|
pub fn DeepSeekV4KeyValueParameterPlan::digest_sha256(self : Self) -> String {
  self.digest_sha256
}

///|
pub fn DeepSeekV4KeyValueParameterPlan::require_interpretation_authority(
  _self : Self,
) -> Unit raise DeepSeekV4KeyValueParameterPlanError {
  raise NoInterpretationAuthority
}

///|
pub fn DeepSeekV4KeyValueParameterPlan::require_materialization_authority(
  _self : Self,
) -> Unit raise DeepSeekV4KeyValueParameterPlanError {
  raise NoMaterializationAuthority
}

///|
pub fn DeepSeekV4KeyValueParameterPlan::require_upload_authority(
  _self : Self,
) -> Unit raise DeepSeekV4KeyValueParameterPlanError {
  raise NoUploadAuthority
}

///|
pub fn DeepSeekV4KeyValueParameterPlan::require_launch_authority(
  _self : Self,
) -> Unit raise DeepSeekV4KeyValueParameterPlanError {
  raise NoLaunchAuthority
}
