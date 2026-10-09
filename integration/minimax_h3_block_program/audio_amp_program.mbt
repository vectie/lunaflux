///|
/// Prepared complete AMP body. Owns function handles and reusable scratch;
/// input/output/checkpoint allocations remain external and queue-leased.
pub struct AudioAmpProgram {
  priv source : @minimax_h3_block_source.AudioAmpSource
  priv functions : Array[@device.Function]
  priv mut scratch : @device.Allocation?
  priv mut layout : @joint_diffusion_device.AudioAmpLayout?
  priv cache : AudioWeightCache
  priv mut cached_parameters : @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout?
}

///|
pub fn AudioAmpProgram::new(
  source : @minimax_h3_block_source.AudioAmpSource,
) -> Self {
  {
    source,
    functions: [],
    scratch: None,
    layout: None,
    cache: AudioWeightCache::new(),
    cached_parameters: None,
  }
}

///|
pub fn AudioAmpProgram::prepare(
  self : Self,
  context : @device.Context,
  aot_module : @device.Module,
  max_scratch_bytes~ : Int64,
) -> Unit raise @device.DeviceError {
  guard self.functions.is_empty() && self.scratch is None else {
    raise @device.DeviceError::InvalidArgument(FunctionLoad)
  }
  let layout = @joint_diffusion_device.AudioAmpLayout::new(
    batch=self.source.batch,
    channels=self.source.channels,
    time=self.source.time,
    max_bytes=max_scratch_bytes,
  )
  acquire_functions(self.source.symbols[:], self.functions, symbol => {
    aot_module.load_function(@utf8.encode(symbol))
  })
  self.scratch = Some(context.allocate(layout.scratch_bytes))
  self.layout = Some(layout)
}

///|
pub fn AudioAmpProgram::bind(
  self : Self,
  parameters : @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout,
  input~ : @device.Allocation,
  input_offset~ : Int64,
  output~ : @device.Allocation,
  output_offset~ : Int64,
  weights~ : @device.Allocation,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.layout is Some(layout) && self.scratch is Some(scratch) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let cache = self.cache.get(weights)
  if cache is Some(_) {
    guard self.cached_parameters is Some(original) &&
      audio_amp_same_parameters(original, parameters) else {
      raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
    }
  }
  layout.launches(
    self.functions,
    parameters,
    input~,
    input_offset~,
    output~,
    output_offset~,
    weights~,
    scratch~,
    normalized_cache?=cache.map(a => (a, audio_amp_cache_offsets(layout).0)),
  )
}

///|
/// Release all consuming queues first. Retain this owner after partial failure.
pub fn AudioAmpProgram::close(self : Self) -> Unit raise @device.DeviceError {
  self.layout = None
  self.cache.close()
  self.cached_parameters = None
  if self.scratch is Some(scratch) {
    scratch.close()
    self.scratch = None
  }
  release_functions(self.functions, function => function.close())
}
