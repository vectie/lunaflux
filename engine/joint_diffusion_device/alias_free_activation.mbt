///|
/// F32 channel-major ratio-two/12-tap alias-free SnakeBeta workspace contract.
/// Both intermediate arrays live together and cannot share storage.
pub struct AliasFreeActivationLayout {
  value_bytes : Int64
  intermediate_bytes : Int64
  activated_offset : Int64
  scratch_bytes : Int64
  parameter_bytes : Int64
}

///|
pub fn AliasFreeActivationLayout::new(
  batch~ : Int,
  channels~ : Int,
  time~ : Int,
  max_scratch_bytes~ : Int64,
) -> Self raise @device.DeviceError {
  guard batch > 0 &&
    channels > 0 &&
    time > 0 &&
    time <= 1073741823 &&
    max_scratch_bytes > 0L else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  let outer = batch.to_int64() * channels.to_int64()
  guard outer <= 2147483647L / time.to_int64() else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  let value = outer * time.to_int64() * 4L
  let intermediate = value * 2L
  let activated = (intermediate + 255L) / 256L * 256L
  guard activated + intermediate <= max_scratch_bytes else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  {
    value_bytes: value,
    intermediate_bytes: intermediate,
    activated_offset: activated,
    scratch_bytes: activated + intermediate,
    parameter_bytes: channels.to_int64() * 4L,
  }
}

///|
/// Bind alias_free_activation_cuda_source's three admitted functions. Scratch
/// must not overlap input/output/weights. Input and output may be identical:
/// input is fully consumed by the first launch before the final launch writes.
pub fn AliasFreeActivationLayout::launches(
  self : Self,
  upsample : @device.Function,
  activation : @device.Function,
  downsample : @device.Function,
  input~ : @device.Allocation,
  input_offset~ : Int64,
  output~ : @device.Allocation,
  output_offset~ : Int64,
  scratch~ : @device.Allocation,
  scratch_offset~ : Int64,
  weights~ : @device.Allocation,
  alpha_offset~ : Int64,
  beta_offset~ : Int64,
  upsample_filter_offset~ : Int64,
  downsample_filter_offset~ : Int64,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard scratch_offset >= 0L &&
    scratch_offset <= scratch.byte_count() &&
    self.scratch_bytes <= scratch.byte_count() - scratch_offset else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  fn region(
    allocation : @device.Allocation,
    offset : Int64,
    bytes : Int64,
  ) -> @device.KernelArgument raise @device.DeviceError {
    @device.KernelArgument::device_region(
      allocation,
      offset~,
      byte_count=bytes,
      alignment=4L,
    )
  }
  let up = region(scratch, scratch_offset, self.intermediate_bytes)
  let activated = region(
    scratch,
    scratch_offset + self.activated_offset,
    self.intermediate_bytes,
  )
  [
    @device.OrderedKernelLaunch::new(
      upsample,
      linear_dimensions(self.intermediate_bytes / 4L, 256),
      @device.KernelArguments::new([
        region(input, input_offset, self.value_bytes),
        region(weights, upsample_filter_offset, 48L),
        up,
      ]),
    ),
    @device.OrderedKernelLaunch::new(
      activation,
      linear_dimensions(self.intermediate_bytes / 4L, 256),
      @device.KernelArguments::new([
        up,
        region(weights, alpha_offset, self.parameter_bytes),
        region(weights, beta_offset, self.parameter_bytes),
        activated,
      ]),
    ),
    @device.OrderedKernelLaunch::new(
      downsample,
      linear_dimensions(self.value_bytes / 4L, 256),
      @device.KernelArguments::new([
        activated,
        region(weights, downsample_filter_offset, 48L),
        region(output, output_offset, self.value_bytes),
      ]),
    ),
  ]
}
