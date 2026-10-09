///|
/// One complete three-branch AMP body: three residual pairs per branch, then
/// ordered three-way averaging. Scratch is shared sequentially across branches.
pub struct AudioAmpLayout {
  activation : AliasFreeActivationLayout
  convolutions : ReadOnlyArray[AudioConvolutionLayout]
  value_offsets : ReadOnlyArray[Int64]
  normalized_offset : Int64
  scratch_bytes : Int64
}

///|
pub fn AudioAmpLayout::new(
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
  let convolutions = [3, 7, 11].map(kernel => {
    AudioConvolutionLayout::new(
      batch~,
      channels~,
      time~,
      kernel~,
      max_weight_bytes=max_bytes,
    )
  })
  fn align(n : Int64) -> Int64 {
    (n + 255L) / 256L * 256L
  }
  let stride = align(activation.value_bytes)
  let start = align(activation.scratch_bytes)
  // activation output, convolution output, residual ping/pong, three branches.
  let offsets = Array::makei(7, i => start + i.to_int64() * stride)
  let normalized = start + 7L * stride
  let total = normalized + convolutions[2].weight_bytes
  guard total <= max_bytes else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  {
    activation,
    convolutions: ReadOnlyArray::from_array(convolutions),
    value_offsets: ReadOnlyArray::from_array(offsets),
    normalized_offset: normalized,
    scratch_bytes: total,
  }
}

///|
/// Functions: three shared activation entries, 18 normalize/convolve pairs
/// in block/pair/side order, residual, average. All functions must match layout.
/// Scratch must be distinct from input, output and checkpoint weights.
pub fn AudioAmpLayout::launches(
  self : Self,
  functions : ArrayView[@device.Function],
  parameters : @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout,
  input~ : @device.Allocation,
  input_offset~ : Int64,
  output~ : @device.Allocation,
  output_offset~ : Int64,
  scratch~ : @device.Allocation,
  weights~ : @device.Allocation,
  normalized_cache? : (@device.Allocation, ReadOnlyArray[Int64]),
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard functions.length() == 41 &&
    scratch.byte_count() >= self.scratch_bytes &&
    weights.byte_count() >= parameters.total_arena_bytes() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  fn parameter(
    role : @joint_audio_amp_parameter_layout.JointAudioAmpParameterRole,
    bytes : Int64,
  ) -> Int64 raise @device.DeviceError {
    for region in parameters.regions() {
      if region.role() == role {
        guard region.byte_count() == bytes else {
          raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
        }
        return region.byte_offset()
      }
    }
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  fn value(
    allocation : @device.Allocation,
    offset : Int64,
  ) -> @device.KernelArgument raise @device.DeviceError {
    @device.KernelArgument::device_region(
      allocation,
      offset~,
      byte_count=self.activation.value_bytes,
      alignment=4L,
    )
  }
  let launches : Array[@device.OrderedKernelLaunch] = []
  if normalized_cache is Some((_, offsets)) && offsets.length() != 18 {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let offsets = self.value_offsets
  for block in 0..<3 {
    let convolution = self.convolutions[block]
    for pair in 0..<3 {
      let (residual, residual_offset) = if pair == 0 {
        (input, input_offset)
      } else {
        (scratch, offsets[2 + (pair - 1) % 2])
      }
      for side in 0..<2 {
        let activation_index = pair * 2 + side
        let (source, source_offset) = if side == 0 {
          (residual, residual_offset)
        } else {
          (scratch, offsets[1])
        }
        launches.append(
          self.activation.launches(
            functions[0],
            functions[1],
            functions[2],
            input=source,
            input_offset=source_offset,
            output=scratch,
            output_offset=offsets[0],
            scratch~,
            scratch_offset=0L,
            weights~,
            alpha_offset=parameter(
              ActivationAlpha(block, activation_index),
              self.activation.parameter_bytes,
            ),
            beta_offset=parameter(
              ActivationBeta(block, activation_index),
              self.activation.parameter_bytes,
            ),
            upsample_filter_offset=parameter(
              ActivationUpsampleFilter(block, activation_index),
              48L,
            ),
            downsample_filter_offset=parameter(
              ActivationDownsampleFilter(block, activation_index),
              48L,
            ),
          ),
        )
        let function_index = 3 + (block * 6 + pair * 2 + side) * 2
        let (normalized, normalized_offset) = match normalized_cache {
          Some((allocation, offsets)) =>
            (allocation, offsets[block * 6 + pair * 2 + side])
          None => (scratch, self.normalized_offset)
        }
        if normalized_cache is None {
          launches.push(
            convolution.normalize_launch(
              functions[function_index],
              weights~,
              g_offset=parameter(
                ConvolutionWeightG(block, pair, side),
                convolution.channel_bytes,
              ),
              v_offset=parameter(
                ConvolutionWeightV(block, pair, side),
                convolution.weight_bytes,
              ),
              normalized=scratch,
              normalized_offset=self.normalized_offset,
            ),
          )
        }
        launches.push(
          convolution.convolve_launch(
            functions[function_index + 1],
            input=scratch,
            input_offset=offsets[0],
            normalized~,
            normalized_offset~,
            bias=weights,
            bias_offset=parameter(
              ConvolutionBias(block, pair, side),
              convolution.channel_bytes,
            ),
            output=scratch,
            output_offset=offsets[1],
          ),
        )
      }
      let destination = if pair == 2 {
        offsets[4 + block]
      } else {
        offsets[2 + pair % 2]
      }
      launches.push(
        @device.OrderedKernelLaunch::new(
          functions[39],
          linear_dimensions(self.activation.value_bytes / 4L, 256),
          @device.KernelArguments::new([
            value(residual, residual_offset),
            value(scratch, offsets[1]),
            value(scratch, destination),
          ]),
        ),
      )
    }
  }
  launches.push(
    @device.OrderedKernelLaunch::new(
      functions[40],
      linear_dimensions(self.activation.value_bytes / 4L, 256),
      @device.KernelArguments::new([
        value(scratch, offsets[4]),
        value(scratch, offsets[5]),
        value(scratch, offsets[6]),
        value(output, output_offset),
      ]),
    ),
  )
  launches
}
