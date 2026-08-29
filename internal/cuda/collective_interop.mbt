///|
/// Forces the CUDA collective-interoperability object into static native
/// links whose later archives consume its lease ABI.
pub fn link_collective_interop() -> Unit {
  raw_collective_interop_link_anchor()
}

///|
/// Captures a vendor-neutral context identity without opening a resource.
pub fn Context::collective_token(
  self : Context,
) -> @device_collective_interop.ContextToken {
  raw_collective_context_token(self)
}

///|
/// Captures a vendor-neutral ordered-queue identity without allocating.
pub fn Stream::collective_token(
  self : Stream,
) -> @device_collective_interop.QueueToken {
  raw_collective_queue_token(self)
}

///|
/// Captures a vendor-neutral allocation identity without allocating.
pub fn Allocation::collective_token(
  self : Allocation,
) -> @device_collective_interop.RegionToken {
  raw_collective_region_token(self)
}
