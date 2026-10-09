///|
/// Time-preserving F32 convolution storage. Dilation/padding
/// are compiled into the admitted kernel; they do not change these extents.
pub struct AudioConvolutionLayout {
  channels : Int
  output_channels : Int
  value_bytes : Int64
  output_bytes : Int64
  weight_bytes : Int64
  channel_bytes : Int64
}

///|
pub fn AudioConvolutionLayout::new(
  batch~ : Int,
  channels~ : Int,
  output_channels? : Int,
  time~ : Int,
  kernel~ : Int,
  max_weight_bytes~ : Int64,
) -> Self raise @device.DeviceError {
  let output_channels = output_channels.unwrap_or(channels)
  guard batch > 0 &&
    channels > 0 &&
    output_channels > 0 &&
    time > 0 &&
    kernel > 1 &&
    max_weight_bytes > 0L else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  let c = channels.to_int64()
  let o = output_channels.to_int64()
  guard batch.to_int64() * c <= 2147483647L / time.to_int64() &&
    batch.to_int64() * o <= 2147483647L / time.to_int64() &&
    o <= max_weight_bytes / 4L / c / kernel.to_int64() else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  {
    channels,
    output_channels,
    value_bytes: batch.to_int64() * c * time.to_int64() * 4L,
    output_bytes: batch.to_int64() * o * time.to_int64() * 4L,
    weight_bytes: o * c * kernel.to_int64() * 4L,
    channel_bytes: o * 4L,
  }
}

///|
/// Weight-only preparation may run once and retain normalized weights across
/// requests. The normalized region must not overlap g/v checkpoint storage.
pub fn AudioConvolutionLayout::normalize_launch(
  self : Self,
  function : @device.Function,
  weights~ : @device.Allocation,
  g_offset~ : Int64,
  v_offset~ : Int64,
  normalized~ : @device.Allocation,
  normalized_offset~ : Int64,
) -> @device.OrderedKernelLaunch raise @device.DeviceError {
  @device.OrderedKernelLaunch::new(
    function,
    @device.KernelLaunchDimensions::new(
      grid_x=self.output_channels,
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
        offset=g_offset,
        byte_count=self.channel_bytes,
        alignment=4L,
      ),
      @device.KernelArgument::device_region(
        weights,
        offset=v_offset,
        byte_count=self.weight_bytes,
        alignment=4L,
      ),
      @device.KernelArgument::device_region(
        normalized,
        offset=normalized_offset,
        byte_count=self.weight_bytes,
        alignment=4L,
      ),
    ]),
  )
}

///|
/// Input/output must not overlap: convolution reads neighboring time/channel
/// cells. Bind after weight normalization, or reuse a completed immutable cache.
pub fn AudioConvolutionLayout::convolve_launch(
  self : Self,
  function : @device.Function,
  input~ : @device.Allocation,
  input_offset~ : Int64,
  normalized~ : @device.Allocation,
  normalized_offset~ : Int64,
  bias~ : @device.Allocation,
  bias_offset~ : Int64,
  output~ : @device.Allocation,
  output_offset~ : Int64,
) -> @device.OrderedKernelLaunch raise @device.DeviceError {
  @device.OrderedKernelLaunch::new(
    function,
    linear_dimensions(self.output_bytes / 4L, 256),
    @device.KernelArguments::new([
      @device.KernelArgument::device_region(
        input,
        offset=input_offset,
        byte_count=self.value_bytes,
        alignment=4L,
      ),
      @device.KernelArgument::device_region(
        normalized,
        offset=normalized_offset,
        byte_count=self.weight_bytes,
        alignment=4L,
      ),
      @device.KernelArgument::device_region(
        bias,
        offset=bias_offset,
        byte_count=self.channel_bytes,
        alignment=4L,
      ),
      @device.KernelArgument::device_region(
        output,
        offset=output_offset,
        byte_count=self.output_bytes,
        alignment=4L,
      ),
    ]),
  )
}
