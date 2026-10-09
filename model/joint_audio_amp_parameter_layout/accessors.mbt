///|
pub fn JointAudioAmpParameterLayout::regions(
  self : Self,
) -> ReadOnlyArray[JointAudioAmpParameterRegion] {
  self.regions
}

///|
pub fn JointAudioAmpParameterLayout::total_arena_bytes(self : Self) -> Int64 {
  self.total_arena_bytes
}

///|
pub fn JointAudioAmpParameterLayout::digest(
  self : Self,
) -> JointAudioAmpParameterLayoutDigest {
  self.digest
}

///|
pub fn JointAudioAmpParameterLayoutDigest::as_hex(self : Self) -> String {
  self.hex
}

///|
pub fn JointAudioAmpParameterRegion::role(
  self : Self,
) -> JointAudioAmpParameterRole {
  self.role
}

///|
pub fn JointAudioAmpParameterRegion::tensor_name(self : Self) -> String {
  self.tensor_name
}

///|
pub fn JointAudioAmpParameterRegion::element_count(self : Self) -> Int64 {
  self.element_count
}

///|
pub fn JointAudioAmpParameterRegion::byte_offset(self : Self) -> Int64 {
  self.byte_offset
}

///|
pub fn JointAudioAmpParameterRegion::byte_count(self : Self) -> Int64 {
  self.byte_count
}
