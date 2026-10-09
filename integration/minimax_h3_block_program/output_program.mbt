///|
/// Final-head owner. Retain after failed prepare and close explicitly; the
/// module and all external tensors remain caller-owned, as with Program.
pub struct OutputProgram {
  priv source : @minimax_h3_block_source.OutputSource
  priv functions : Array[@device.Function]
  priv mut workspace : @device.Allocation?
  priv mut layout : OutputLayout?
}

///|
pub fn OutputProgram::new(
  source : @minimax_h3_block_source.OutputSource,
) -> Self {
  { source, functions: [], workspace: None, layout: None, }
}

///|
pub fn OutputProgram::prepare(
  self : Self,
  context : @device.Context,
  aot_module : @device.Module,
  max_workspace_bytes~ : Int64,
) -> Unit raise @device.DeviceError {
  guard self.functions.is_empty() && self.workspace is None else {
    raise @device.DeviceError::InvalidArgument(FunctionLoad)
  }
  let s = self.source
  let layout = output_layout(
    s.packed_rows,
    s.model.denoiser().hidden_size(),
    s.distinct_timesteps,
    s.video_rows,
    s.audio_rows,
    max_workspace_bytes,
  )
  acquire_functions(s.symbols[:], self.functions, name => {
    aot_module.load_function(@utf8.encode(name))
  })
  self.workspace = Some(context.allocate(layout.bytes))
  self.layout = Some(layout)
}

///|
pub fn OutputProgram::close(self : Self) -> Unit raise @device.DeviceError {
  self.layout = None
  if self.workspace is Some(workspace) {
    workspace.close()
    self.workspace = None
  }
  release_functions(self.functions, function => function.close())
}

///|
/// Prepare six ordered launches; this never submits work or updates latents.
/// Bind once for each timestep-index pattern, execute only after the denoiser,
/// and pass this complete sequence before both flow updates.
pub fn OutputProgram::bind(
  self : Self,
  storage : @joint_diffusion_device.LatentStorage,
  modulation : @joint_diffusion_device.ModulationBindings,
  packing : @minimax_h3_weights.OutputWeightPacking,
  conditioning~ : @device.Allocation,
  conditioning_offset~ : Int64,
  state~ : @device.Allocation,
  state_offset~ : Int64,
  weights~ : @device.Allocation,
) -> Array[@device.OrderedKernelLaunch] raise {
  guard self.layout is Some(layout) &&
    self.workspace is Some(workspace) &&
    modulation.matches_geometry(
      rows=self.source.packed_rows,
      width=self.source.model.denoiser().hidden_size(),
      parameter_rows=self.source.distinct_timesteps,
    ) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let s = self.source
  storage.validate_prediction_rows(
    s.model,
    video_rows=s.video_rows,
    audio_rows=s.audio_rows,
  )
  let shape = s.model.denoiser()
  let f = self.functions
  let parameters = @joint_diffusion_device.ParameterProjectionLayout::new(
    rows=s.distinct_timesteps,
    input_width=shape.final_adaln_conditioning_dimension(),
    width=shape.hidden_size(),
    groups=1,
    fields=2,
  ).launch(
    f[0],
    input=conditioning,
    input_offset=conditioning_offset,
    weights~,
    weight_offset=packing.adaln_offset,
    bias_offset=packing.adaln_bias_offset,
    output=workspace,
    output_offset=layout.parameters,
  )
  let norm = @joint_diffusion_device.block_rms_norm_launch(
    f[1],
    rows=s.packed_rows,
    width=shape.hidden_size(),
    input=state,
    input_offset=state_offset,
    weights~,
    weight_offset=packing.norm_offset,
    output=workspace,
    output_offset=0L,
  )
  let video_mod = modulation.selected_output_launch(
    f[2],
    start_row=s.video_start,
    selected_rows=s.video_rows,
    normalized=workspace,
    normalized_offset=0L,
    parameters=workspace,
    parameter_offset=layout.parameters,
    output=workspace,
    output_offset=layout.selected,
  )
  let video = storage.output_projection_launch(
    f[4],
    s.model,
    latent=VideoLatent,
    hidden=workspace,
    hidden_offset=layout.selected,
    weights~,
    weight_offset=packing.video_offset,
    bias=weights,
    bias_offset=packing.video_bias_offset,
  )
  let audio_mod = modulation.selected_output_launch(
    f[3],
    start_row=s.audio_start,
    selected_rows=s.audio_rows,
    normalized=workspace,
    normalized_offset=0L,
    parameters=workspace,
    parameter_offset=layout.parameters,
    output=workspace,
    output_offset=layout.selected,
  )
  let audio = storage.output_projection_launch(
    f[5],
    s.model,
    latent=AudioLatent,
    hidden=workspace,
    hidden_offset=layout.selected,
    weights~,
    weight_offset=packing.audio_offset,
    bias=weights,
    bias_offset=packing.audio_bias_offset,
  )
  [parameters, norm, video_mod, video, audio_mod, audio]
}
