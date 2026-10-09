///|
/// Startup-only normalization; independent of activation input/output lengths.
pub fn AudioUpsampleLayout::normalize_launch(
  self : Self,
  function : @device.Function,
  weights : @device.Allocation,
  normalized : @device.Allocation,
) -> @device.OrderedKernelLaunch raise @device.DeviceError {
  @device.OrderedKernelLaunch::new(
    function,
    @device.KernelLaunchDimensions::new(
      grid_x=self.input_channels,
      grid_y=1,
      grid_z=1,
      block_x=1,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    @device.KernelArguments::new([
      @device.KernelArgument::device_region(
        weights,
        offset=0L,
        byte_count=self.g_bytes,
        alignment=4L,
      ),
      @device.KernelArgument::device_region(
        weights,
        offset=self.g_bytes,
        byte_count=self.weight_bytes,
        alignment=4L,
      ),
      @device.KernelArgument::device_region(
        normalized,
        offset=0L,
        byte_count=self.weight_bytes,
        alignment=4L,
      ),
    ]),
  )
}
