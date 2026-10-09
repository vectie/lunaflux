///|
/// Build once before execution. Parameters and workspace are reusable across
/// serialized blocks; concurrent frames must use separate Program owners.
/// Conditioning is already SiLU-activated BF16, shared across all blocks.
pub fn Program::bind(
  self : Self,
  rotary : @joint_diffusion_device.RotaryBindings,
  modulation : @joint_diffusion_device.ModulationBindings,
  packing : @minimax_h3_weights.BlockWeightPacking,
  conditioning~ : @device.Allocation,
  conditioning_offset~ : Int64,
  state~ : @device.Allocation,
  state_offset~ : Int64,
  weights~ : @device.Allocation,
  parameters~ : @device.Allocation,
  parameter_offset~ : Int64,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.ready &&
    self.workspace is Some(workspace) &&
    modulation.parameter_row_count() == self.source.distinct_timesteps * 3 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let f = self.functions
  let attention = @joint_diffusion_device.AttentionFunctions::new(
    f[4],
    f[5],
    f[6],
    f[7],
  )
  let functions = @joint_diffusion_device.BlockFunctions::new(
    attention,
    norm=f[1],
    affine=f[2],
    residual=f[3],
    gate_up=f[8],
    down=f[9],
  )
  workspace.bind_conditioned_block(
    functions,
    f[0],
    rotary,
    modulation,
    conditioning~,
    conditioning_offset~,
    conditioning_width=self.source.shape.final_adaln_conditioning_dimension(),
    state~,
    state_offset~,
    weights~,
    attention_weight_offset=packing.attention_offset,
    mlp_weight_offset=packing.mlp_offset,
    norm_weight_offset=packing.norm_offset,
    parameter_weight_offset=packing.adaln_offset,
    parameter_bias_offset=packing.adaln_bias_offset,
    parameters~,
    parameter_offset~,
    modality_count=3,
  )
}

///|
/// Shared activation runs once per distinct timestep table, not once per block.
pub fn Program::conditioning_launch(
  self : Self,
  input~ : @device.Allocation,
  input_offset~ : Int64,
  output~ : @device.Allocation,
  output_offset~ : Int64,
) -> @device.OrderedKernelLaunch raise @device.DeviceError {
  guard self.ready else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  @joint_diffusion_device.conditioning_activation_launch(
    self.functions[10],
    elements=self.source.distinct_timesteps.to_int64() *
      self.source.shape.final_adaln_conditioning_dimension().to_int64(),
    input~,
    input_offset~,
    output~,
    output_offset~,
  )
}
