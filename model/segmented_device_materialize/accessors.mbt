///|
pub fn SegmentedHostSourceDigest::as_hex(self : Self) -> String {
  self.hex
}

///|
pub fn SegmentedDeviceLayoutDigest::as_hex(self : Self) -> String {
  self.hex
}

///|
pub fn SegmentedDeviceTensorRegion::tensor_ordinal(self : Self) -> Int {
  self.tensor_ordinal
}

///|
pub fn SegmentedDeviceTensorRegion::source_arena_ordinal(self : Self) -> Int {
  self.source_arena_ordinal
}

///|
pub fn SegmentedDeviceTensorRegion::source_offset(self : Self) -> Int {
  self.source_offset
}

///|
pub fn SegmentedDeviceTensorRegion::device_offset(self : Self) -> Int64 {
  self.device_offset
}

///|
pub fn SegmentedDeviceTensorRegion::byte_count(self : Self) -> Int64 {
  self.byte_count
}

///|
pub fn SegmentedDeviceLayout::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn SegmentedDeviceLayout::source_digest(
  self : Self,
) -> SegmentedHostSourceDigest {
  self.source_digest
}

///|
pub fn SegmentedDeviceLayout::digest(
  self : Self,
) -> SegmentedDeviceLayoutDigest {
  self.digest
}

///|
pub fn SegmentedDeviceLayout::alignment_bytes(self : Self) -> Int64 {
  self.alignment_bytes
}

///|
pub fn SegmentedDeviceLayout::device_arena_bytes(self : Self) -> Int64 {
  self.device_arena_bytes
}

///|
pub fn SegmentedDeviceLayout::materialized_bytes(self : Self) -> Int64 {
  self.materialized_bytes
}

///|
pub fn SegmentedDeviceLayout::source_arena_count(self : Self) -> Int {
  self.source_arena_lengths.length()
}

///|
pub fn SegmentedDeviceLayout::source_arena_length(
  self : Self,
  index : Int,
) -> Int? {
  self.source_arena_lengths.get(index)
}

///|
pub fn SegmentedDeviceLayout::region_count(self : Self) -> Int {
  self.regions.length()
}

///|
pub fn SegmentedDeviceLayout::region(
  self : Self,
  region_ordinal : Int,
) -> SegmentedDeviceTensorRegion? {
  self.regions.get(region_ordinal)
}

///|
pub fn SegmentedDeviceWeights::layout(self : Self) -> SegmentedDeviceLayout {
  self.layout
}

///|
pub fn SegmentedDeviceWeights::allocation(self : Self) -> @device.Allocation {
  self.allocation
}

///|
pub fn SegmentedDeviceWeights::close(
  self : Self,
) -> Unit raise SegmentedDeviceMaterializationError {
  self.allocation.close() catch {
    error => raise DeviceFailed(error)
  }
}

///|
pub fn FailedSegmentedDevicePreparation::cause(
  self : Self,
) -> SegmentedDeviceMaterializationError {
  self.cause
}

///|
pub fn FailedSegmentedDevicePreparation::stage(
  self : Self,
) -> SegmentedDeviceFailureStage {
  self.stage
}

///|
pub fn FailedSegmentedDevicePreparation::cleanup_failure(
  self : Self,
) -> @device.DeviceError {
  self.cleanup_failure
}

///|
pub fn FailedSegmentedDevicePreparation::is_closed(self : Self) -> Bool {
  !self.open
}

///|
pub fn FailedSegmentedDevicePreparation::retry_close(
  self : Self,
) -> Unit raise SegmentedDeviceMaterializationError {
  if !self.open {
    return
  }
  self.allocation.close() catch {
    error => {
      self.cleanup_failure = error
      raise CleanupFailed(stage=self.stage, error)
    }
  }
  self.open = false
}
