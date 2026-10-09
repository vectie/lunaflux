///|
/// Preparation-time borrow of a typed latent region. This does not assert GPU
/// completion; downstream launches must be ordered after the region's producer.
/// Queues retain allocation leases and must close before this storage owner.
pub fn[T] LatentStorage::with_region(
  self : Self,
  buffer : @joint_diffusion_loop.LatentBuffer,
  consume : (@device.Allocation, Int64, Int64) -> T raise @device.DeviceError,
) -> T raise @device.DeviceError {
  let region = self.plan.region(buffer)
  consume(self.allocation, region.offset, region.byte_count)
}

///|
pub fn LatentStorage::matches_prepared(
  self : Self,
  plan : @joint_diffusion_loop.PreparedPlan,
) -> Bool {
  self.plan.matches_prepared(plan)
}
