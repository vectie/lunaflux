///|
/// Bind cosine_sine_timestep_rows_source before the MLP's first stage.
/// Input contains the distinct unscaled F32 timesteps; frequency rows are F32
/// with cosine followed by sine, as required by the shared source renderer.
pub fn TimestepMlpLayout::frequency_launch(
  self : Self,
  function : @device.Function,
  timesteps~ : @device.Allocation,
  timestep_offset~ : Int64,
  output~ : @device.Allocation,
  output_offset~ : Int64,
) -> @device.OrderedKernelLaunch raise @device.DeviceError {
  guard self.input_width % 2 == 0 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let elements = self.rows.to_int64() * self.input_width.to_int64()
  @device.OrderedKernelLaunch::new(
    function,
    linear_dimensions(elements, 256),
    @device.KernelArguments::new([
      @device.KernelArgument::device_region(
        timesteps,
        offset=timestep_offset,
        byte_count=self.rows.to_int64() * 4L,
        alignment=4L,
      ),
      @device.KernelArgument::device_region(
        output,
        offset=output_offset,
        byte_count=elements * 4L,
        alignment=4L,
      ),
    ]),
  )
}
