///|
/// Normalize once before preparing any request. Budget covers the aggregate
/// retained cache, not a separate ceiling reused independently for each stage.
/// Partial failures retain stage cache owners; close stages to clean up.
pub fn AudioDecoder::prepare_weights(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  parameters : ArrayView[
    (
      @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout,
      @device.Allocation,
      @device.Allocation,
    ),
  ],
  frontend_weights~ : @device.Allocation,
  tail_weights~ : @device.Allocation,
  max_cache_bytes~ : Int64,
) -> Unit raise @device.DeviceError {
  self.lease.require_idle()
  guard !self.closed &&
    !self.chain.closed &&
    parameters.length() == 7 &&
    self.frontend.layout is Some(front) &&
    self.tail.layout is Some(tail) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let budgets : Array[Int64] = [front.convolution.weight_bytes]
  for stage in self.chain.stages {
    guard stage.layout is Some(up) && stage.amp.layout is Some(amp) else {
      raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
    }
    budgets.push(up.upsample.weight_bytes + audio_amp_cache_offsets(amp).1)
  }
  budgets.push(tail.convolution.weight_bytes)
  audio_cache_budget(budgets, max_cache_bytes)
  self.frontend.prepare_weights(
    context,
    stream,
    frontend_weights,
    max_cache_bytes=budgets[0],
  )
  for i, stage in self.chain.stages {
    let (layout, up_weights, amp_weights) = parameters[i]
    stage.prepare_weights(
      context,
      stream,
      layout,
      up_weights,
      amp_weights,
      max_cache_bytes=budgets[i + 1],
    )
  }
  self.tail.prepare_weights(
    context,
    stream,
    tail_weights,
    max_cache_bytes=budgets[8],
  )
}

///|
fn audio_cache_budget(
  parts : ArrayView[Int64],
  maximum : Int64,
) -> Unit raise @device.DeviceError {
  guard maximum > 0L else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  for bytes in parts; remaining = maximum {
    guard bytes > 0L && bytes <= remaining else {
      raise @device.DeviceError::InvalidArgument(AllocationCreate)
    }
    continue remaining - bytes
  } nobreak {
    ()
  }
}

///|
fn AudioDecoder::require_cached_weights(
  self : Self,
) -> Unit raise @device.DeviceError {
  guard self.frontend.cache.ready && self.tail.cache.ready else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  for stage in self.chain.stages {
    guard stage.cache.ready && stage.amp.cache.ready else {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
    }
  }
}
