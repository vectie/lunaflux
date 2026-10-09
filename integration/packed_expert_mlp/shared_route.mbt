///|
/// Initialize the shared expert's all-row, unit-score route once at startup.
/// Buffers are reused across steps; only counts[3] changes with live rows.
pub fn initialize_shared_route(
  context : @device.Context,
  plan : @precision.ExpertMlpPrecision,
  indices~ : @device.Allocation,
  weights~ : @device.Allocation,
  expert_map~ : @device.Allocation,
) -> Unit raise @device.DeviceError {
  let (rows, _, _, experts, selected) = plan.geometry()
  guard experts == 1 &&
    selected == 1 &&
    plan.local_experts().length() == 1 &&
    plan.local_experts()[0] == 0 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let bytes = rows.to_int64() * 4L
  guard bytes <= 2147483647L else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let buffer = FixedArray::make(bytes.to_int(), b'\x00')
  context.copy_from_fixed_host(indices, buffer, 0L, 0L, bytes)
  context.copy_from_fixed_host(expert_map, buffer, 0L, 0L, 4L)
  for row in 0..<rows {
    buffer[row * 4 + 2] = b'\x80'
    buffer[row * 4 + 3] = b'\x3f'
  }
  context.copy_from_fixed_host(weights, buffer, 0L, 0L, bytes)
}
