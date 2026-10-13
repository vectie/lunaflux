///|
/// The queue is the effect boundary: every old-state read retires before the
/// final stage commits. Scratch is allocated once and owned by this program.
fn RecurrentDeltaProgram::append_delta_launches(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  launches : Array[@device.OrderedKernelLaunch],
  arguments : Array[@device.KernelArgument],
) -> Unit raise @device.DeviceError {
  let bytes = self.source.workspace_bytes()
  if bytes > 0L {
    let scratch = context.allocate(bytes)
    self.intermediates.push(scratch)
    arguments.push(region(scratch, bytes, 4L))
  }
  for stage in 0..<self.source.stage_count() {
    let function = aot.load_function(
      @utf8.encode(self.source.stage_symbol(stage)),
    )
    self.functions.push(function)
    let (grid_x, grid_y, grid_z, block_x) = self.source.stage_launch(stage)
    launches.push(
      @device.OrderedKernelLaunch::new(
        function,
        @device.KernelLaunchDimensions::new(
          grid_x~,
          grid_y~,
          grid_z~,
          block_x~,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        @device.KernelArguments::new(arguments),
      ),
    )
  }
}
