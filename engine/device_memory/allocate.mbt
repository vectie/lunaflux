///|
fn[T] allocate_owned(
  byte_count : Int64,
  open : (Int64) -> T raise @device.DeviceError,
  size : (T) -> Int64,
  close : (T) -> Unit raise @device.DeviceError,
) -> T raise DeviceMemoryError {
  let resource = open(byte_count) catch { error => raise DeviceFailed(error) }
  if size(resource) != byte_count {
    close(resource) catch {
      error => raise CleanupFailed(AllocationValidation, error)
    }
    raise AllocationSizeMismatch
  }
  resource
}

///|
/// Allocates the complete activation/workspace arena exactly once at startup.
///
/// No token-step path is present in this package. The caller owns the returned
/// arena and must close it before closing `context`.
pub fn allocate(
  context : @device.Context,
  memory_plan : DeviceMemoryPlan,
) -> DeviceMemoryArena raise DeviceMemoryError {
  let allocation = allocate_owned(
    memory_plan.arena_bytes(),
    bytes => context.allocate(bytes),
    resource => resource.byte_count(),
    resource => resource.close(),
  )
  { allocation, plan: memory_plan }
}
