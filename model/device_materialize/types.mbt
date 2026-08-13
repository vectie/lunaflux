///|
/// Final device-arena placement for one semantic model tensor.
pub struct DeviceWeightRegion {
  reference : @plan.TensorRef
  offset_bytes : Int64
  byte_length : Int64
} derive(Debug, Eq)

///|
/// Deterministic immutable layout for one complete model weight arena.
pub struct DeviceWeightLayout {
  identity : @spec.ModelIdentity
  alignment_bytes : Int64
  arena_bytes : Int64
  materialized_bytes : Int64
  regions : ReadOnlyArray[DeviceWeightRegion]
} derive(Debug, Eq)

///|
/// Lifecycle owner of one completely loaded device-weight arena.
///
/// Call `close` before closing its context. The allocation returned by
/// `allocation` aliases the same safe handle: closing either handle closes the
/// arena for every alias. Callers should therefore keep lifecycle authority in
/// `DeviceWeights` and use the alias only while dispatching device operations.
pub struct DeviceWeights {
  priv allocation : @device.Allocation
  layout : DeviceWeightLayout
}

///|
/// Explicit lifecycle result for device-weight loading.
///
/// The cleanup-required branch is returned only when either a primary load
/// failure or terminal source-file close failure is followed by failure of the
/// first deterministic allocation close.
pub(all) enum DeviceWeightPreparation {
  WeightsReady(DeviceWeights)
  WeightsCleanupRequired(FailedDeviceWeightPreparation)
}

///|
/// Cleanup authority retained when allocation close fails after a primary load
/// failure or after terminal source-file close prevents readiness. The
/// incomplete allocation is never exposed.
pub struct FailedDeviceWeightPreparation {
  priv allocation : @device.Allocation
  priv cause : DeviceWeightError
  priv stage : DeviceWeightFailureStage
  priv mut cleanup_failure : @device.DeviceError
  priv mut open : Bool
}

///|
pub fn DeviceWeightRegion::reference(
  self : DeviceWeightRegion,
) -> @plan.TensorRef {
  self.reference
}

///|
pub fn DeviceWeightRegion::offset_bytes(self : DeviceWeightRegion) -> Int64 {
  self.offset_bytes
}

///|
pub fn DeviceWeightRegion::byte_length(self : DeviceWeightRegion) -> Int64 {
  self.byte_length
}

///|
pub fn DeviceWeightRegion::end_offset_bytes(self : DeviceWeightRegion) -> Int64 {
  self.offset_bytes + self.byte_length
}

///|
pub fn DeviceWeightLayout::identity(
  self : DeviceWeightLayout,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn DeviceWeightLayout::alignment_bytes(self : DeviceWeightLayout) -> Int64 {
  self.alignment_bytes
}

///|
pub fn DeviceWeightLayout::arena_bytes(self : DeviceWeightLayout) -> Int64 {
  self.arena_bytes
}

///|
pub fn DeviceWeightLayout::materialized_bytes(
  self : DeviceWeightLayout,
) -> Int64 {
  self.materialized_bytes
}

///|
pub fn DeviceWeightLayout::length(self : DeviceWeightLayout) -> Int {
  self.regions.length()
}

///|
pub fn DeviceWeightLayout::region(
  self : DeviceWeightLayout,
  reference : @plan.TensorRef,
) -> DeviceWeightRegion? {
  self.regions.get(reference.as_int())
}

///|
pub fn DeviceWeights::identity(self : DeviceWeights) -> @spec.ModelIdentity {
  self.layout.identity()
}

///|
pub fn DeviceWeights::layout(self : DeviceWeights) -> DeviceWeightLayout {
  self.layout
}

///|
pub fn DeviceWeights::region(
  self : DeviceWeights,
  reference : @plan.TensorRef,
) -> DeviceWeightRegion? {
  self.layout.region(reference)
}

///|
/// Returns an alias of the safe opaque allocation for device execution.
/// Closing this alias also closes the arena represented by `DeviceWeights`.
pub fn DeviceWeights::allocation(self : DeviceWeights) -> @device.Allocation {
  self.allocation
}

///|
pub fn DeviceWeights::close(
  self : DeviceWeights,
) -> Unit raise DeviceWeightError {
  self.allocation.close() catch {
    error => raise DeviceFailed(error)
  }
}

///|
pub fn FailedDeviceWeightPreparation::cause(
  self : FailedDeviceWeightPreparation,
) -> DeviceWeightError {
  self.cause
}

///|
pub fn FailedDeviceWeightPreparation::stage(
  self : FailedDeviceWeightPreparation,
) -> DeviceWeightFailureStage {
  self.stage
}

///|
pub fn FailedDeviceWeightPreparation::cleanup_failure(
  self : FailedDeviceWeightPreparation,
) -> @device.DeviceError {
  self.cleanup_failure
}

///|
pub fn FailedDeviceWeightPreparation::is_closed(
  self : FailedDeviceWeightPreparation,
) -> Bool {
  !self.open
}

///|
/// Retries release of the retained incomplete allocation. Success and later
/// calls are idempotent; failure preserves authority for another retry.
pub fn FailedDeviceWeightPreparation::retry_close(
  self : FailedDeviceWeightPreparation,
) -> Unit raise DeviceWeightError {
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
