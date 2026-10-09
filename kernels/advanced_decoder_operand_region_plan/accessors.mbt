///|
pub fn AdvancedDecoderOperandRegionPlan::status(
  _self : Self,
) -> AdvancedDecoderOperandRegionPlanStatus {
  InertUnqualified
}

///|
pub fn AdvancedDecoderOperandRegionPlan::regions(
  self : Self,
) -> ReadOnlyArray[AdvancedDecoderOperandRegion] {
  self.regions
}

///|
pub fn AdvancedDecoderOperandRegionPlan::region(
  self : Self,
  operand_ordinal : Int,
) -> AdvancedDecoderOperandRegion? {
  self.regions.get(operand_ordinal)
}

///|
pub fn AdvancedDecoderOperandRegionPlan::digest(
  self : Self,
) -> AdvancedDecoderOperandRegionPlanDigest {
  self.digest
}

///|
pub fn AdvancedDecoderOperandRegionPlanDigest::as_hex(self : Self) -> String {
  let AdvancedDecoderOperandRegionPlanDigest(hex) = self
  hex
}

///|
pub fn AdvancedDecoderOperandRegion::operand_ordinal(self : Self) -> Int {
  self.operand_ordinal
}

///|
pub fn AdvancedDecoderOperandRegion::role(
  self : Self,
) -> @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperandRole {
  self.role
}

///|
pub fn AdvancedDecoderOperandRegion::byte_count(self : Self) -> Int64 {
  self.byte_count
}

///|
pub fn AdvancedDecoderOperandRegion::alignment(self : Self) -> Int64 {
  self.alignment
}

///|
pub fn AdvancedDecoderOperandRegion::allocation_identity_sha256(
  self : Self,
) -> String {
  self.allocation_identity_sha256
}

///|
pub fn AdvancedDecoderOperandRegion::device_offset(self : Self) -> Int64 {
  self.device_offset
}

///|
pub fn AdvancedDecoderOperandRegion::end_offset(self : Self) -> Int64 {
  self.end_offset
}

///|
pub fn AdvancedDecoderOperandRegion::allocation_capacity_bytes(
  self : Self,
) -> Int64 {
  self.allocation_capacity_bytes
}

///|
pub fn AdvancedDecoderOperandRegionPlan::require_launch_authority(
  _self : Self,
) -> Unit raise AdvancedDecoderOperandRegionPlanError {
  raise LaunchAuthorityUnavailable
}
