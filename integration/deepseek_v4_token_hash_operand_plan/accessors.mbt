///|
pub fn DeepSeekV4TokenHashOperandPlan::status(
  _self : Self,
) -> DeepSeekV4TokenHashOperandPlanStatus {
  InertUnqualified
}

///|
pub fn DeepSeekV4TokenHashOperandPlan::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn DeepSeekV4TokenHashOperandPlan::requirements_digest(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn DeepSeekV4TokenHashOperandPlan::requirement_ordinal(self : Self) -> Int {
  self.requirement_ordinal
}

///|
pub fn DeepSeekV4TokenHashOperandPlan::layer_index(self : Self) -> Int {
  self.layer_index
}

///|
pub fn DeepSeekV4TokenHashOperandPlan::routed_expert_count(self : Self) -> Int {
  self.routed_expert_count
}

///|
pub fn DeepSeekV4TokenHashOperandPlan::operand_ordinal(self : Self) -> Int {
  self.operand_ordinal
}

///|
pub fn DeepSeekV4TokenHashOperandPlan::operand_role(
  self : Self,
) -> @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperandRole {
  self.operand_role
}

///|
pub fn DeepSeekV4TokenHashOperandPlan::byte_count(self : Self) -> Int64 {
  self.byte_count
}

///|
pub fn DeepSeekV4TokenHashOperandPlan::alignment(self : Self) -> Int64 {
  self.alignment
}

///|
pub fn DeepSeekV4TokenHashOperandPlan::region_ordinal(self : Self) -> Int {
  self.region_ordinal
}

///|
pub fn DeepSeekV4TokenHashOperandPlan::region_tensor_ordinal(
  self : Self,
) -> Int {
  self.region_tensor_ordinal
}

///|
pub fn DeepSeekV4TokenHashOperandPlan::device_offset(self : Self) -> Int64 {
  self.device_offset
}

///|
pub fn DeepSeekV4TokenHashOperandPlan::sidecar_layout_digest(
  self : Self,
) -> @segmented_device_materialize.SegmentedDeviceLayoutDigest {
  self.sidecar_layout_digest
}

///|
pub fn DeepSeekV4TokenHashOperandPlan::digest(
  self : Self,
) -> DeepSeekV4TokenHashOperandPlanDigest {
  self.digest
}

///|
pub fn DeepSeekV4TokenHashOperandPlanDigest::as_hex(self : Self) -> String {
  let DeepSeekV4TokenHashOperandPlanDigest(hex) = self
  hex
}

///|
pub fn DeepSeekV4TokenHashOperandPlan::require_launch_authority(
  _self : Self,
) -> Unit raise DeepSeekV4TokenHashOperandPlanError {
  raise NoLaunchAuthority
}
