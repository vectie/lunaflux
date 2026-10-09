///|
/// Generic pointwise projection followed by a time-preserving normalized conv.
pub struct AudioFrontendLayout {
  convolution : AudioConvolutionLayout
  input_bytes : Int64
  projection_weight_bytes : Int64
  projection_bias_bytes : Int64
  g_offset : Int64
  v_offset : Int64
  bias_offset : Int64
  weight_bytes : Int64
  normalized_offset : Int64
  scratch_bytes : Int64
}

///|
pub fn AudioFrontendLayout::new(
  batch~ : Int,
  input_channels~ : Int,
  hidden_channels~ : Int,
  output_channels~ : Int,
  time~ : Int,
  kernel~ : Int,
  max_bytes~ : Int64,
) -> Self raise @device.DeviceError {
  guard input_channels > 0 &&
    batch > 0 &&
    time > 0 &&
    batch.to_int64() * input_channels.to_int64() <=
    2147483647L / time.to_int64() else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  let convolution = AudioConvolutionLayout::new(
    batch~,
    channels=hidden_channels,
    output_channels~,
    time~,
    kernel~,
    max_weight_bytes=max_bytes,
  )
  guard input_channels.to_int64() <= max_bytes / 4L / hidden_channels.to_int64() else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  let projection_weight_bytes = input_channels.to_int64() *
    hidden_channels.to_int64() *
    4L
  let projection_bias_bytes = hidden_channels.to_int64() * 4L
  let g_offset = projection_weight_bytes + projection_bias_bytes
  let v_offset = g_offset + convolution.channel_bytes
  let bias_offset = v_offset + convolution.weight_bytes
  let weight_bytes = bias_offset + convolution.channel_bytes
  let normalized_offset = (convolution.value_bytes + 255L) / 256L * 256L
  let scratch_bytes = normalized_offset + convolution.weight_bytes
  guard scratch_bytes <= max_bytes && weight_bytes <= max_bytes else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  {
    convolution,
    input_bytes: batch.to_int64() *
    input_channels.to_int64() *
    time.to_int64() *
    4L,
    projection_weight_bytes,
    projection_bias_bytes,
    g_offset,
    v_offset,
    bias_offset,
    weight_bytes,
    normalized_offset,
    scratch_bytes,
  }
}

///|
/// Scratch must be distinct from live input, output and immutable weights.
pub fn AudioFrontendLayout::launches(
  self : Self,
  functions : ArrayView[@device.Function],
  input~ : @device.Allocation,
  input_offset~ : Int64,
  output~ : @device.Allocation,
  output_offset~ : Int64,
  scratch~ : @device.Allocation,
  weights~ : @device.Allocation,
  normalized_cache? : @device.Allocation,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard functions.length() == 3 &&
    scratch.byte_count() >= self.scratch_bytes &&
    weights.byte_count() >= self.weight_bytes else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let project = @device.OrderedKernelLaunch::new(
    functions[0],
    linear_dimensions(self.convolution.value_bytes / 4L, 256),
    @device.KernelArguments::new([
      @device.KernelArgument::device_region(
        input,
        offset=input_offset,
        byte_count=self.input_bytes,
        alignment=4L,
      ),
      @device.KernelArgument::device_region(
        weights,
        offset=0L,
        byte_count=self.projection_weight_bytes,
        alignment=4L,
      ),
      @device.KernelArgument::device_region(
        weights,
        offset=self.projection_weight_bytes,
        byte_count=self.projection_bias_bytes,
        alignment=4L,
      ),
      @device.KernelArgument::device_region(
        scratch,
        offset=0L,
        byte_count=self.convolution.value_bytes,
        alignment=4L,
      ),
    ]),
  )
  let launches = [project]
  if normalized_cache is None {
    launches.push(
      self.convolution.normalize_launch(
        functions[1],
        weights~,
        g_offset=self.g_offset,
        v_offset=self.v_offset,
        normalized=scratch,
        normalized_offset=self.normalized_offset,
      ),
    )
  }
  launches.push(
    self.convolution.convolve_launch(
      functions[2],
      input=scratch,
      input_offset=0L,
      normalized=normalized_cache.unwrap_or(scratch),
      normalized_offset=if normalized_cache is Some(_) {
        0L
      } else {
        self.normalized_offset
      },
      bias=weights,
      bias_offset=self.bias_offset,
      output~,
      output_offset~,
    ),
  )
  launches
}
