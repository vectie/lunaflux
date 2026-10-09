///|
pub struct InputProgram {
  priv source : @minimax_h3_block_source.InputSource
  priv functions : Array[@device.Function]
  priv mut projections : @joint_diffusion_device.LatentProjections?
  priv mut packed : @joint_diffusion_device.PackedHidden?
  priv mut ready : Bool
}

///|
pub fn InputProgram::new(source : @minimax_h3_block_source.InputSource) -> Self {
  { source, functions: [], projections: None, packed: None, ready: false, }
}

///|
/// Retain the owner on any partial failure and close before retrying. Mapping
/// upload happens once, before frames can borrow the packed state.
pub fn InputProgram::prepare(
  self : Self,
  context : @device.Context,
  aot_module : @device.Module,
  storage : @joint_diffusion_device.LatentStorage,
  max_projection_bytes~ : Int64,
) -> Unit raise {
  guard self.functions.is_empty() &&
    self.projections is None &&
    self.packed is None else {
    raise @device.DeviceError::InvalidArgument(FunctionLoad)
  }
  let plan = self.source.sequence.assembly()
  storage.validate_prediction_rows(
    self.source.model,
    video_rows=plan.input_rows(Video),
    audio_rows=plan.input_rows(Audio),
  )
  acquire_functions(self.source.symbols[:], self.functions, name => {
    aot_module.load_function(@utf8.encode(name))
  })
  self.projections = Some(
    @joint_diffusion_device.LatentProjections::allocate(
      context,
      storage,
      self.source.model,
      max_bytes=max_projection_bytes,
    ),
  )
  let packed = @joint_diffusion_device.PackedHidden::allocate(context, plan)
  self.packed = Some(packed)
  packed.upload_mapping()
  self.ready = true
}

///|
/// Borrowed state owner for stack binding; do not close it independently.
pub fn InputProgram::packed_state(
  self : Self,
) -> @joint_diffusion_device.PackedHidden raise @device.DeviceError {
  guard self.ready && self.packed is Some(packed) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  packed
}

///|
/// Prepared input projections must read the exact arena subsequently updated
/// by this request, not another allocation with an identical shape or plan.
pub fn InputProgram::matches_storage(
  self : Self,
  storage : @joint_diffusion_device.LatentStorage,
) -> Bool {
  self.ready &&
  self.projections is Some(projections) &&
  projections.matches_storage(storage)
}

///|
pub fn InputProgram::close(self : Self) -> Unit raise @device.DeviceError {
  self.ready = false
  if self.packed is Some(packed) {
    packed.close()
    self.packed = None
  }
  if self.projections is Some(projections) {
    projections.close()
    self.projections = None
  }
  release_functions(self.functions, function => function.close())
}

///|
/// Conditioning contains already encoded/projected text and reference rows,
/// not timestep embeddings. This list reads current latents anew every step.
pub fn InputProgram::bind(
  self : Self,
  packing : @minimax_h3_weights.InputWeightPacking,
  weights~ : @device.Allocation,
  conditioning~ : @device.Allocation?,
  conditioning_offset~ : Int64,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.ready &&
    self.projections is Some(projections) &&
    self.packed is Some(packed) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let video = projections.launch(
    self.functions[0],
    latent=VideoLatent,
    weights~,
    weight_offset=packing.video_offset,
    bias=weights,
    bias_offset=packing.video_bias_offset,
  )
  let audio = projections.launch(
    self.functions[1],
    latent=AudioLatent,
    weights~,
    weight_offset=packing.audio_offset,
    bias=weights,
    bias_offset=packing.audio_bias_offset,
  )
  let gather = packed.launch(
    self.functions[2],
    projections,
    conditioning~,
    conditioning_offset~,
    block_size=256,
  )
  [video, audio, gather]
}
