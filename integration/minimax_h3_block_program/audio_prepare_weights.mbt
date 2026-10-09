///|
/// Normalize immutable AMP checkpoint operands once at model startup. The
/// caller must keep checkpoint contents immutable for the prepared lifetime.
/// Cache budget is additional to existing activation/scratch storage: the
/// uncached normalization scratch is retained to preserve prepare/bind ABI.
pub fn AudioAmpProgram::prepare_weights(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  parameters : @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout,
  weights : @device.Allocation,
  max_cache_bytes~ : Int64,
) -> Unit raise @device.DeviceError {
  guard self.layout is Some(layout) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let (offsets, bytes) = audio_amp_cache_offsets(layout)
  self.cache.prepare(context, stream, bytes, max_cache_bytes, weights, normalized => {
    fn parameter(
      role : @joint_audio_amp_parameter_layout.JointAudioAmpParameterRole,
      bytes : Int64,
    ) -> Int64 raise @device.DeviceError {
      for region in parameters.regions() {
        if region.role() == role && region.byte_count() == bytes {
          return region.byte_offset()
        }
      }
      raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
    }
    let launches = []
    for block in 0..<3 {
      let conv = layout.convolutions[block]
      for pair in 0..<3 {
        for side in 0..<2 {
          let i = block * 6 + pair * 2 + side
          launches.push(
            conv.normalize_launch(
              self.functions[3 + i * 2],
              weights~,
              g_offset=parameter(
                ConvolutionWeightG(block, pair, side),
                conv.channel_bytes,
              ),
              v_offset=parameter(
                ConvolutionWeightV(block, pair, side),
                conv.weight_bytes,
              ),
              normalized~,
              normalized_offset=offsets[i],
            ),
          )
        }
      }
    }
    launches
  })
  self.cached_parameters = Some(parameters)
}

///|
pub fn AudioFrontendProgram::prepare_weights(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  weights : @device.Allocation,
  max_cache_bytes~ : Int64,
) -> Unit raise @device.DeviceError {
  guard self.layout is Some(layout) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  self.cache.prepare(
    context,
    stream,
    layout.convolution.weight_bytes,
    max_cache_bytes,
    weights,
    normalized => {
      [
        layout.convolution.normalize_launch(
          self.functions[1],
          weights~,
          g_offset=layout.g_offset,
          v_offset=layout.v_offset,
          normalized~,
          normalized_offset=0L,
        ),
      ]
    },
  )
}

///|
pub fn AudioTailProgram::prepare_weights(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  weights : @device.Allocation,
  max_cache_bytes~ : Int64,
) -> Unit raise @device.DeviceError {
  guard self.layout is Some(layout) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let g = layout.activation.parameter_bytes * 2L + 96L
  self.cache.prepare(
    context,
    stream,
    layout.convolution.weight_bytes,
    max_cache_bytes,
    weights,
    normalized => {
      [
        layout.convolution.normalize_launch(
          self.functions[3],
          weights~,
          g_offset=g,
          v_offset=g + 4L,
          normalized~,
          normalized_offset=0L,
        ),
      ]
    },
  )
}

///|
pub fn AudioDecodeStage::prepare_weights(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  parameters : @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout,
  up_weights : @device.Allocation,
  amp_weights : @device.Allocation,
  max_cache_bytes~ : Int64,
) -> Unit raise @device.DeviceError {
  guard self.layout is Some(layout) && self.workspace is Some(_) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let u = layout.upsample
  guard self.amp.layout is Some(amp_layout) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let amp_bytes = audio_amp_cache_offsets(amp_layout).1
  guard u.weight_bytes <= max_cache_bytes &&
    amp_bytes <= max_cache_bytes - u.weight_bytes else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  self.cache.prepare(
    context,
    stream,
    u.weight_bytes,
    max_cache_bytes,
    up_weights,
    normalized => {
      [u.normalize_launch(self.functions[0], up_weights, normalized)]
    },
  )
  self.amp.prepare_weights(
    context,
    stream,
    parameters,
    amp_weights,
    max_cache_bytes=max_cache_bytes - u.weight_bytes,
  )
}
