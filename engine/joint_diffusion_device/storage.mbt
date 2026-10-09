///|
/// Owns a single request-local device allocation, with no GC-only release.
/// Allocation and coefficient upload are separate so a failed copy never loses
/// the owner needed for deterministic cleanup. Contents begin uninitialized.
pub struct LatentStorage {
  priv plan : @joint_diffusion_loop.LatentStoragePlan
  priv allocation : @device.Allocation
}

///|
pub fn LatentStorage::allocate(
  context : @device.Context,
  plan : @joint_diffusion_loop.LatentStoragePlan,
) -> Self raise @device.DeviceError {
  { plan, allocation: context.allocate(plan.byte_count()), }
}

///|
/// Borrow an exact region for startup kernel binding. Queue preparation owns
/// its allocation lease; this storage must outlive all such queues.
pub fn LatentStorage::argument(
  self : Self,
  buffer : @joint_diffusion_loop.LatentBuffer,
) -> @device.KernelArgument raise @device.DeviceError {
  let region = self.plan.region(buffer)
  @device.KernelArgument::device_region(
    self.allocation,
    offset=region.offset,
    byte_count=region.byte_count,
    alignment=256L,
  )
}

///|
/// One startup upload before any frame is submitted. Subsequent frames borrow
/// fixed offsets; do not call again while a queue references this allocation.
pub fn LatentStorage::upload_flow_table(
  self : Self,
  context : @device.Context,
) -> FlowBindings raise @device.DeviceError {
  FlowBindings::upload(
    context,
    self.plan.flow_table(),
    self.allocation,
    offset=self.plan.region(FlowTableBuffer).offset,
  )
}

///|
/// Explicit startup input or terminal output transfer, never a denoise-step
/// operation. Exact sizes prevent truncating a stereo or patch-major tensor.
pub fn LatentStorage::copy_from_host(
  self : Self,
  buffer : @joint_diffusion_loop.LatentBuffer,
  bytes : Bytes,
) -> Unit raise @device.DeviceError {
  let region = self.plan.region(buffer)
  guard bytes.length().to_int64() == region.byte_count else {
    raise @device.DeviceError::InvalidArgument(CopyToDevice)
  }
  self.allocation.copy_from_host(bytes, 0L, region.offset, region.byte_count)
}

///|
pub fn LatentStorage::copy_to_host(
  self : Self,
  buffer : @joint_diffusion_loop.LatentBuffer,
) -> Bytes raise @device.DeviceError {
  let region = self.plan.region(buffer)
  self.allocation.copy_to_host(region.offset, region.byte_count)
}

///|
/// Close only after frame queues release their leases. Busy/error retains the
/// owner; callers must retry cleanup rather than drop it.
pub fn LatentStorage::close(self : Self) -> Unit raise @device.DeviceError {
  self.allocation.close()
}
