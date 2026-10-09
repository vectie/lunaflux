///|
pub struct AudioTailLayout {
  activation : AliasFreeActivationLayout
  convolution : AudioConvolutionLayout
  activated_offset : Int64
  normalized_offset : Int64
  scratch_bytes : Int64
  weight_bytes : Int64
}

///|
pub fn AudioTailLayout::new(
  batch~ : Int,
  channels~ : Int,
  time~ : Int,
  max_bytes~ : Int64,
) -> Self raise @device.DeviceError {
  let activation = AliasFreeActivationLayout::new(
    batch~,
    channels~,
    time~,
    max_scratch_bytes=max_bytes,
  )
  let convolution = AudioConvolutionLayout::new(
    batch~,
    channels~,
    output_channels=1,
    time~,
    kernel=7,
    max_weight_bytes=max_bytes,
  )
  let activated_offset = (activation.scratch_bytes + 255L) / 256L * 256L
  let normalized_offset = (activated_offset + activation.value_bytes + 255L) /
    256L *
    256L
  let scratch_bytes = normalized_offset + convolution.weight_bytes
  guard scratch_bytes <= max_bytes else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  {
    activation,
    convolution,
    activated_offset,
    normalized_offset,
    scratch_bytes,
    weight_bytes: activation.parameter_bytes * 2L +
    96L +
    4L +
    convolution.weight_bytes,
  }
}

///|
/// Weights: contiguous F32 alpha, beta, up/down filters, conv g/v.
/// Official H3 uses neither bias nor final tanh. Scratch is separate from
/// input/output/weights; the unused bias ABI operand borrows g without reading it.
pub fn AudioTailLayout::launches(
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
  guard functions.length() == 5 &&
    scratch.byte_count() >= self.scratch_bytes &&
    weights.byte_count() >= self.weight_bytes else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let c = self.activation.parameter_bytes
  let g = c * 2L + 96L
  let v = g + 4L
  let launches = self.activation.launches(
    functions[0],
    functions[1],
    functions[2],
    input~,
    input_offset~,
    output=scratch,
    output_offset=self.activated_offset,
    scratch~,
    scratch_offset=0L,
    weights~,
    alpha_offset=0L,
    beta_offset=c,
    upsample_filter_offset=c * 2L,
    downsample_filter_offset=c * 2L + 48L,
  )
  if normalized_cache is None {
    launches.push(
      self.convolution.normalize_launch(
        functions[3],
        weights~,
        g_offset=g,
        v_offset=v,
        normalized=scratch,
        normalized_offset=self.normalized_offset,
      ),
    )
  }
  launches.push(
    self.convolution.convolve_launch(
      functions[4],
      input=scratch,
      input_offset=self.activated_offset,
      normalized=normalized_cache.unwrap_or(scratch),
      normalized_offset=if normalized_cache is Some(_) {
        0L
      } else {
        self.normalized_offset
      },
      bias=weights,
      bias_offset=g,
      output~,
      output_offset~,
    ),
  )
  launches
}
