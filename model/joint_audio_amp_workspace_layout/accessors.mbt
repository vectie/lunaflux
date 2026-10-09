///|
pub fn JointAudioAmpWorkspaceLayout::stages(
  self : Self,
) -> ReadOnlyArray[JointAudioAmpExecutionStage] {
  self.stages
}

///|
pub fn JointAudioAmpWorkspaceLayout::regions(
  self : Self,
) -> ReadOnlyArray[JointAudioAmpWorkspaceRegion] {
  self.regions
}

///|
pub fn JointAudioAmpWorkspaceLayout::workspace_bytes(self : Self) -> Int64 {
  self.workspace_bytes
}

///|
pub fn JointAudioAmpWorkspaceLayout::digest(
  self : Self,
) -> JointAudioAmpWorkspaceLayoutDigest {
  self.digest
}

///|
pub fn JointAudioAmpWorkspaceLayoutDigest::as_hex(self : Self) -> String {
  self.hex
}

///|
pub fn JointAudioAmpWorkspaceRegion::role(
  self : Self,
) -> JointAudioAmpWorkspaceRole {
  self.role
}

///|
pub fn JointAudioAmpWorkspaceRegion::storage_class(
  self : Self,
) -> JointAudioAmpStorageClass {
  self.storage_class
}

///|
pub fn JointAudioAmpWorkspaceRegion::byte_offset(self : Self) -> Int64 {
  self.byte_offset
}

///|
pub fn JointAudioAmpWorkspaceRegion::byte_count(self : Self) -> Int64 {
  self.byte_count
}

///|
pub fn JointAudioAmpWorkspaceRegion::live_start(self : Self) -> Int {
  self.live_start
}

///|
pub fn JointAudioAmpWorkspaceRegion::live_end(self : Self) -> Int {
  self.live_end
}
