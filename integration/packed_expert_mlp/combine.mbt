///|
/// Startup binding. The routed input is the F32 reduction destination; shared
/// output is rank-replicated, computed with the same MLP path at score one.
pub fn bind_combine(
  source : @combine.MoeCombineSource,
  function : @device.Function,
  counts~ : @device.Allocation,
  routed~ : @device.Allocation,
  shared~ : @device.Allocation,
  output~ : @device.Allocation,
) -> @device.OrderedKernelLaunch raise @device.DeviceError {
  let (input_bytes, output_bytes) = source.plan().buffer_bytes()
  let (grid, block) = source.launch().dimensions()
  @device.OrderedKernelLaunch::new(
    function,
    @device.KernelLaunchDimensions::new(
      grid_x=grid,
      grid_y=1,
      grid_z=1,
      block_x=block,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    @device.KernelArguments::new([
      region(counts, 20L, 4L),
      region(routed, input_bytes, 4L),
      region(shared, input_bytes, 4L),
      region(output, output_bytes, 2L),
    ]),
  )
}
