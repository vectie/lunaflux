///|
/// Binds a transfer owner without exposing the arena allocation. The caller
/// retains the returned cache, including failed preparation, and closes it
/// before this arena. Native leases enforce that close order.
pub fn DeviceKvArena::prepare_host_cache(
  self : DeviceKvArena,
  context : @device.Context,
  plan : @host_cache.Plan,
) -> @host_cache.Cache raise DeviceKvArenaError {
  require_open(self.lifecycle)
  if context.ordinal() != self.context_ordinal {
    raise WrongContext
  }
  if plan.layout() != self.plan.layout {
    raise InvalidLayout(RegionIndex)
  }
  @host_cache.Cache::prepare(context, self.allocation, plan) catch {
    error => raise DeviceFailed(AllocationCreate, error)
  }
}
