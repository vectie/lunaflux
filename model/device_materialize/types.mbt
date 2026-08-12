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
