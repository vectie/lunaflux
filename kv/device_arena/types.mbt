///|
/// Semantically validated startup description for one persistent KV arena.
///
/// This value owns no native resource. Its opaque construction proves the
/// model identity, canonical V1 byte envelope, and 256-byte region alignment
/// before device allocation may begin.
pub struct DeviceKvArenaPlan {
  priv layout : @device_layout.DeviceKvLayout
}

///|
/// Explicit owner of one persistent device allocation for the complete KV
/// layout. No allocation alias or raw handle is exposed.
///
/// `context` is borrowed during construction and physical-region validation.
/// The caller must keep that exact context open until this owner is closed.
/// The arena and any returned `DeviceKvRegion` must remain confined to the
/// worker that owns the context; regions carry byte metadata, not lifecycle
/// or mutation authority.
pub struct DeviceKvArena {
  priv allocation : @device.Allocation
  priv plan : DeviceKvArenaPlan
  priv context_ordinal : Int
  priv mut lifecycle : DeviceKvArenaLifecycle
}

///|
/// Explicit result after native allocation begins. If post-allocation
/// validation and its first cleanup attempt both fail, `CleanupRequired`
/// preserves the only release authority instead of abandoning it to GC.
pub(all) enum DeviceKvArenaPreparation {
  Ready(DeviceKvArena)
  CleanupRequired(FailedDeviceKvArenaPreparation)
}

///|
/// Opaque owner retained after construction cleanup fails.
pub struct FailedDeviceKvArenaPreparation {
  priv allocation : @device.Allocation
  priv cause : DeviceKvArenaError
  priv mut cleanup_failure : DeviceKvArenaError
  priv mut lifecycle : DeviceKvArenaLifecycle
}

///|
priv enum DeviceKvArenaLifecycle {
  Open
  ClosingFailed
  ClosedState
} derive(Eq)

///|
pub fn DeviceKvArenaPlan::layout(
  self : DeviceKvArenaPlan,
) -> @device_layout.DeviceKvLayout {
  self.layout
}

///|
pub fn DeviceKvArenaPlan::identity(
  self : DeviceKvArenaPlan,
) -> @spec.ModelIdentity {
  self.layout.identity()
}

///|
pub fn DeviceKvArenaPlan::arena_bytes(self : DeviceKvArenaPlan) -> Int64 {
  self.layout.arena_bytes()
}

///|
pub fn DeviceKvArenaPlan::segment_alignment_bytes(
  self : DeviceKvArenaPlan,
) -> Int64 {
  self.layout.segment_alignment_bytes()
}

///|
pub fn DeviceKvArena::plan(self : DeviceKvArena) -> DeviceKvArenaPlan {
  self.plan
}

///|
pub fn DeviceKvArena::layout(
  self : DeviceKvArena,
) -> @device_layout.DeviceKvLayout {
  self.plan.layout()
}

///|
pub fn DeviceKvArena::identity(self : DeviceKvArena) -> @spec.ModelIdentity {
  self.plan.identity()
}

///|
pub fn DeviceKvArena::arena_bytes(self : DeviceKvArena) -> Int64 {
  self.plan.arena_bytes()
}

///|
pub fn DeviceKvArena::context_ordinal(self : DeviceKvArena) -> Int {
  self.context_ordinal
}

///|
pub fn DeviceKvArena::is_closed(self : DeviceKvArena) -> Bool {
  self.lifecycle == ClosedState
}

///|
pub fn DeviceKvArena::is_usable(self : DeviceKvArena) -> Bool {
  self.lifecycle == Open
}

///|
pub fn FailedDeviceKvArenaPreparation::cause(
  self : FailedDeviceKvArenaPreparation,
) -> DeviceKvArenaError {
  self.cause
}

///|
pub fn FailedDeviceKvArenaPreparation::cleanup_failure(
  self : FailedDeviceKvArenaPreparation,
) -> DeviceKvArenaError {
  self.cleanup_failure
}

///|
pub fn FailedDeviceKvArenaPreparation::is_closed(
  self : FailedDeviceKvArenaPreparation,
) -> Bool {
  self.lifecycle == ClosedState
}
