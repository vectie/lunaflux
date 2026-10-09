///|
/// Packed diffusion rows -> channel-major latent -> destandardized VAE input.
/// The permutation and channel geometry are pure request preparation data.
pub struct VaeInputLayout {
  value_bytes : Int64
  parameter_bytes : Int64
}

///|
pub fn VaeInputLayout::new(
  permutation : @dense_permutation_plan.Permutation,
  batch~ : Int,
  channels~ : Int,
  elements_per_channel~ : Int64,
  max_bytes~ : Int64,
) -> Self raise @device.DeviceError {
  guard batch > 0 &&
    channels > 0 &&
    elements_per_channel > 0L &&
    max_bytes > 0L &&
    elements_per_channel <=
    max_bytes / 4L / batch.to_int64() / channels.to_int64() else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  let elements = batch.to_int64() * channels.to_int64() * elements_per_channel
  guard permutation.element_count() == elements else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  { value_bytes: elements * 4L, parameter_bytes: channels.to_int64() * 4L, }
}

///|
/// Bind matching dense permutation (256 threads) and channel-major affine AOT
/// functions. Scratch and output must be disjoint from the live packed input;
/// means/stds are immutable, retained device allocations, never host round trips.
pub fn VaeInputLayout::launches(
  self : Self,
  permutation : @device.Function,
  destandardize : @device.Function,
  input~ : @device.Allocation,
  input_offset~ : Int64,
  scratch~ : @device.Allocation,
  scratch_offset~ : Int64,
  output~ : @device.Allocation,
  output_offset~ : Int64,
  standard_deviations~ : @device.Allocation,
  standard_deviations_offset~ : Int64,
  means~ : @device.Allocation,
  means_offset~ : Int64,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  let input_arg = @device.KernelArgument::device_region(
    input,
    offset=input_offset,
    byte_count=self.value_bytes,
    alignment=4L,
  )
  let scratch_arg = @device.KernelArgument::device_region(
    scratch,
    offset=scratch_offset,
    byte_count=self.value_bytes,
    alignment=4L,
  )
  let output_arg = @device.KernelArgument::device_region(
    output,
    offset=output_offset,
    byte_count=self.value_bytes,
    alignment=4L,
  )
  fn disjoint(
    a : @device.Allocation,
    a_offset : Int64,
    a_bytes : Int64,
    b : @device.Allocation,
    b_offset : Int64,
    b_bytes : Int64,
  ) -> Unit raise @device.DeviceError {
    guard !physical_equal(a, b) ||
      !vae_regions_overlap(a_offset, a_bytes, b_offset, b_bytes) else {
      raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
    }
  }
  disjoint(
    input,
    input_offset,
    self.value_bytes,
    scratch,
    scratch_offset,
    self.value_bytes,
  )
  disjoint(
    input,
    input_offset,
    self.value_bytes,
    output,
    output_offset,
    self.value_bytes,
  )
  disjoint(
    scratch,
    scratch_offset,
    self.value_bytes,
    output,
    output_offset,
    self.value_bytes,
  )
  disjoint(
    standard_deviations,
    standard_deviations_offset,
    self.parameter_bytes,
    scratch,
    scratch_offset,
    self.value_bytes,
  )
  disjoint(
    standard_deviations,
    standard_deviations_offset,
    self.parameter_bytes,
    output,
    output_offset,
    self.value_bytes,
  )
  disjoint(
    means,
    means_offset,
    self.parameter_bytes,
    scratch,
    scratch_offset,
    self.value_bytes,
  )
  disjoint(
    means,
    means_offset,
    self.parameter_bytes,
    output,
    output_offset,
    self.value_bytes,
  )
  let dimensions = linear_dimensions(self.value_bytes / 4L, 256)
  [
    @device.OrderedKernelLaunch::new(
      permutation,
      dimensions,
      @device.KernelArguments::new([input_arg, scratch_arg]),
    ),
    @device.OrderedKernelLaunch::new(
      destandardize,
      dimensions,
      @device.KernelArguments::new([
        scratch_arg,
        @device.KernelArgument::device_region(
          standard_deviations,
          offset=standard_deviations_offset,
          byte_count=self.parameter_bytes,
          alignment=4L,
        ),
        @device.KernelArgument::device_region(
          means,
          offset=means_offset,
          byte_count=self.parameter_bytes,
          alignment=4L,
        ),
        output_arg,
      ]),
    ),
  ]
}

///|
/// Subtraction avoids overflowing an end-offset sum. Negative offsets will
/// also be rejected by device_region; treat them as overlapping here.
fn vae_regions_overlap(
  a : Int64,
  a_bytes : Int64,
  b : Int64,
  b_bytes : Int64,
) -> Bool {
  if a < 0L || b < 0L {
    return true
  }
  if a <= b {
    b - a < a_bytes
  } else {
    a - b < b_bytes
  }
}
