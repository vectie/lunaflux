///|
pub struct AudioFrontendProgram {
  priv source : @minimax_h3_block_source.AudioFrontendSource
  priv functions : Array[@device.Function]
  priv mut scratch : @device.Allocation?
  priv mut layout : @joint_diffusion_device.AudioFrontendLayout?
  priv cache : AudioWeightCache
  priv composition : AudioDecoderLease
}

///|
pub fn AudioFrontendProgram::new(
  source : @minimax_h3_block_source.AudioFrontendSource,
) -> Self {
  {
    source,
    functions: [],
    scratch: None,
    layout: None,
    cache: AudioWeightCache::new(),
    composition: { held: false, },
  }
}

///|
pub fn AudioFrontendProgram::prepare(
  self : Self,
  context : @device.Context,
  aot_module : @device.Module,
  max_scratch_bytes~ : Int64,
) -> Unit raise @device.DeviceError {
  guard self.functions.is_empty() && self.scratch is None else {
    raise @device.DeviceError::InvalidArgument(FunctionLoad)
  }
  let layout = @joint_diffusion_device.AudioFrontendLayout::new(
    batch=self.source.batch,
    input_channels=32,
    hidden_channels=2048,
    output_channels=1024,
    time=self.source.time,
    kernel=7,
    max_bytes=max_scratch_bytes,
  )
  acquire_functions(self.source.symbols[:], self.functions, name => {
    aot_module.load_function(@utf8.encode(name))
  })
  self.scratch = Some(context.allocate(layout.scratch_bytes))
  self.layout = Some(layout)
}

///|
pub fn AudioFrontendProgram::bind(
  self : Self,
  input~ : @device.Allocation,
  input_offset~ : Int64,
  output~ : @device.Allocation,
  output_offset~ : Int64,
  weights~ : @device.Allocation,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.layout is Some(layout) && self.scratch is Some(scratch) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  layout.launches(
    self.functions,
    input~,
    input_offset~,
    output~,
    output_offset~,
    weights~,
    scratch~,
    normalized_cache?=self.cache.get(weights),
  )
}

///|
/// Close consuming queues first; retain owners when release fails for retry.
pub fn AudioFrontendProgram::close(
  self : Self,
) -> Unit raise @device.DeviceError {
  self.composition.require_idle()
  self.layout = None
  self.cache.close()
  if self.scratch is Some(scratch) {
    scratch.close()
    self.scratch = None
  }
  release_functions(self.functions, function => function.close())
}
