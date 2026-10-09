///|
pub struct RefinerProgram {
  priv source : @minimax_h3_block_source.RefinerSource
  priv functions : Array[@device.Function]
  priv mut workspace : @joint_diffusion_device.BlockWorkspace?
  priv mut ready : Bool
}

///|
pub fn RefinerProgram::new(
  source : @minimax_h3_block_source.RefinerSource,
) -> Self {
  { source, functions: [], workspace: None, ready: false, }
}

///|
pub fn RefinerProgram::prepare(
  self : Self,
  context : @device.Context,
  aot_module : @device.Module,
  max_workspace_bytes~ : Int64,
) -> Unit raise @device.DeviceError {
  guard self.functions.is_empty() && self.workspace is None else {
    raise @device.DeviceError::InvalidArgument(FunctionLoad)
  }
  let s = self.source.model.denoiser()
  let layout = @joint_diffusion_device.BlockWorkspaceLayout::new(
    rows=self.source.rows,
    hidden_width=s.hidden_size(),
    head_count=s.attention_head_count(),
    head_dimension=s.attention_head_dimension(),
    intermediate_width=s.feed_forward_dimension(),
    max_bytes=max_workspace_bytes,
  )
  acquire_functions(self.source.symbols[:], self.functions, name => {
    aot_module.load_function(@utf8.encode(name))
  })
  self.workspace = Some(
    @joint_diffusion_device.BlockWorkspace::allocate(context, layout),
  )
  self.ready = true
}

///|
pub fn RefinerProgram::close(self : Self) -> Unit raise @device.DeviceError {
  self.ready = false
  if self.workspace is Some(workspace) {
    workspace.close()
    self.workspace = None
  }
  release_functions(self.functions, function => function.close())
}

///|
/// Refine only live projected text rows, before assembling multimodal state.
/// Residuals and final RMSNorm update that text slice in place. Two official
/// layers share one workspace and are followed by one final normalization.
pub fn RefinerProgram::bind(
  self : Self,
  layers : ArrayView[
    (@minimax_h3_weights.RefinerWeightPacking, @device.Allocation),
  ],
  state~ : @device.Allocation,
  state_offset~ : Int64,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.ready && self.workspace is Some(workspace) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  validate_layer_order(
    layers.map(pair => pair.0.layer),
    self.source.model.denoiser().refiner_layer_count(),
  )
  let f = self.functions
  let attention = @joint_diffusion_device.AttentionFunctions::new(
    f[3],
    f[4],
    f[5],
    f[6],
  )
  let launches : Array[@device.OrderedKernelLaunch] = []
  for (p, weights) in layers {
    launches.append(
      workspace.bind_refiner(
        attention,
        norm=f[0],
        residual=f[2],
        gate_up=f[7],
        down=f[8],
        state~,
        state_offset~,
        weights~,
        attention_weight_offset=p.attention_offset,
        mlp_weight_offset=p.mlp_offset,
        norm_weight_offset=p.norm_offset,
      ),
    )
  }
  let (last, weights) = layers[layers.length() - 1]
  guard last.final_norm_offset is Some(weight_offset) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  launches.push(
    @joint_diffusion_device.block_rms_norm_launch(
      f[1],
      rows=self.source.rows,
      width=self.source.model.denoiser().hidden_size(),
      input=state,
      input_offset=state_offset,
      weights~,
      weight_offset~,
      output=state,
      output_offset=state_offset,
    ),
  )
  launches
}
