///|
/// Bounded structural rule rejected before persistent device allocation.
pub(all) enum DeviceKvArenaRule {
  Identity
  LayoutVersion
  SegmentAlignment
  PageLogicalBytes
  PageStride
  ComponentStride
  LayerStride
  ArenaBytes
  ArenaLimit
  FirstRegion
  LastRegion
  RegionIndex
} derive(Debug, Eq)

///|
/// Stable stage for a failure translated from the public device facade.
pub(all) enum DeviceKvArenaStage {
  AllocationCreate
  AllocationValidation
  RegionValidation
  AllocationClose
} derive(Debug, Eq)

///|
/// Payload-safe failures from persistent device-KV ownership.
///
/// Values contain only fixed rules, scalar indices, and the public device
/// facade's bounded error vocabulary. They contain no pointers, paths, model
/// payloads, allocation handles, or vendor diagnostics.
pub(all) suberror DeviceKvArenaError {
  InvalidLayout(DeviceKvArenaRule)
  AllocationSizeMismatch
  WrongContext
  ArenaPoisoned
  Closed
  DeviceFailed(DeviceKvArenaStage, @device.DeviceError)
} derive(Debug, Eq)
