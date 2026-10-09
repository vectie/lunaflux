///|
pub struct AudioUpsampleLayout {
  input_channels : Int
  output_time : Int
  input_bytes : Int64
  output_bytes : Int64
  weight_bytes : Int64
  g_bytes : Int64
  bias_bytes : Int64
}

///|
pub fn AudioUpsampleLayout::new(
  batch~ : Int,
  input_channels~ : Int,
  output_channels~ : Int,
  input_time~ : Int,
  kernel~ : Int,
  stride~ : Int,
  padding~ : Int,
  max_weight_bytes~ : Int64,
) -> Self raise @device.DeviceError {
  guard batch > 0 &&
    input_channels > 0 &&
    output_channels > 0 &&
    input_time > 0 &&
    kernel > 0 &&
    stride > 0 &&
    padding >= 0 &&
    max_weight_bytes > 0L else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  let output_time = (input_time.to_int64() - 1L) * stride.to_int64() +
    kernel.to_int64() -
    2L * padding.to_int64()
  guard output_time > 0L && output_time <= 2147483647L else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  let ci = input_channels.to_int64()
  let co = output_channels.to_int64()
  guard batch.to_int64() * ci <= 2147483647L / input_time.to_int64() &&
    batch.to_int64() * co <= 2147483647L / output_time &&
    ci <= max_weight_bytes / 4L / co / kernel.to_int64() else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  {
    input_channels,
    output_time: output_time.to_int(),
    input_bytes: batch.to_int64() * ci * input_time.to_int64() * 4L,
    output_bytes: batch.to_int64() * co * output_time * 4L,
    weight_bytes: ci * co * kernel.to_int64() * 4L,
    g_bytes: ci * 4L,
    bias_bytes: co * 4L,
  }
}

///|
/// Bind the F32 weight-normalized transposed-convolution ABI. Normalization
/// is per INPUT channel, unlike ordinary convolution's per-output normalization.
/// All writable regions must be separate from live input and checkpoint data.
pub fn AudioUpsampleLayout::launches(
  self : Self,
  normalize : @device.Function,
  convolve : @device.Function,
  input~ : @device.Allocation,
  input_offset~ : Int64,
  output~ : @device.Allocation,
  output_offset~ : Int64,
  normalized~ : @device.Allocation,
  normalized_offset~ : Int64,
  weights~ : @device.Allocation,
  g_offset~ : Int64,
  v_offset~ : Int64,
  bias_offset~ : Int64,
  cached? : Bool = false,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  fn region(
    a : @device.Allocation,
    offset : Int64,
    bytes : Int64,
  ) -> @device.KernelArgument raise @device.DeviceError {
    @device.KernelArgument::device_region(
      a,
      offset~,
      byte_count=bytes,
      alignment=4L,
    )
  }
  let normalized_argument = region(
    normalized,
    normalized_offset,
    self.weight_bytes,
  )
  let launches : Array[@device.OrderedKernelLaunch] = []
  if !cached {
    launches.push(
      @device.OrderedKernelLaunch::new(
        normalize,
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
          region(weights, g_offset, self.g_bytes),
          region(weights, v_offset, self.weight_bytes),
          normalized_argument,
        ]),
      ),
    )
  }
  launches.push(
    @device.OrderedKernelLaunch::new(
      convolve,
      linear_dimensions(self.output_bytes / 4L, 256),
      @device.KernelArguments::new([
        region(input, input_offset, self.input_bytes),
        normalized_argument,
        region(weights, bias_offset, self.bias_bytes),
        region(output, output_offset, self.output_bytes),
      ]),
    ),
  )
  launches
}
