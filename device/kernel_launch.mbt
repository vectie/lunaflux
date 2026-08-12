///|
/// Launches this AOT function and synchronizes `stream` before returning.
/// `arguments` is immutable and reusable, so this method performs no
/// steady-state MoonBit allocation or payload conversion.
pub fn Function::launch_synchronous(
  self : Function,
  dimensions : KernelLaunchDimensions,
  arguments : KernelArguments,
  stream : Stream,
) -> Unit raise DeviceError {
  self.raw.launch_synchronous(
    stream.raw,
    dimensions.grid_x,
    dimensions.grid_y,
    dimensions.grid_z,
    dimensions.block_x,
    dimensions.block_y,
    dimensions.block_z,
    dimensions.shared_memory_bytes,
    arguments.allocations,
    arguments.offsets,
    arguments.byte_counts,
    arguments.alignments,
  ) catch {
    error => raise error_from_cuda(error)
  }
}
