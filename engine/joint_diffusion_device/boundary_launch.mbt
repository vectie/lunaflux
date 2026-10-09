///|
/// Bind the admitted u32-cell permutation function for this exact boundary
/// map and block size. The plan supplies both non-overlapping regions, so the
/// caller cannot accidentally request an in-place transpose. Preparation only:
/// no upload, scratch allocation, tensor copy or kernel submission occurs here.
pub fn LatentStorage::boundary_launch(
  self : Self,
  function : @device.Function,
  boundary~ : @joint_diffusion_loop.LatentBoundary,
  block_size~ : Int,
) -> @device.OrderedKernelLaunch raise @device.DeviceError {
  let transfer = self.plan.boundary(boundary)
  let dimensions = linear_dimensions(
    transfer.permutation.element_count(),
    block_size,
  )
  let arguments = @device.KernelArguments::new([
    self.argument(transfer.input),
    self.argument(transfer.output),
  ])
  @device.OrderedKernelLaunch::new(function, dimensions, arguments)
}

///|
/// Shared CUDA linear launch geometry. Pure arithmetic before queue creation;
/// keep the renderer's Int32 element bound and avoid addition overflow.
fn linear_dimensions(
  elements : Int64,
  block_size : Int,
) -> @device.KernelLaunchDimensions raise @device.DeviceError {
  guard elements > 0L &&
    elements <= 2147483647L &&
    block_size > 0 &&
    block_size <= 1024 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  @device.KernelLaunchDimensions::new(
    grid_x=((elements - 1L) / block_size.to_int64() + 1L).to_int(),
    grid_y=1,
    grid_z=1,
    block_x=block_size,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
}
