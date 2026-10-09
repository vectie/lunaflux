///|
struct TimestepWorkspace {
  mlp : @joint_diffusion_device.TimestepMlpLayout
  hidden : Int64
  output : Int64
  bytes : Int64
}

///|
fn timestep_workspace(
  rows : Int,
  input : Int,
  hidden : Int,
  output : Int,
  budget : Int64,
) -> TimestepWorkspace raise @device.DeviceError {
  let mlp = @joint_diffusion_device.TimestepMlpLayout::new(
    rows~,
    input_width=input,
    hidden_width=hidden,
    output_width=output,
  )
  fn align(n : Int64) -> Int64 {
    n + (256L - n % 256L) % 256L
  }
  let hidden_offset = align(rows.to_int64() * input.to_int64() * 4L)
  let output_offset = hidden_offset + align(mlp.scratch_bytes)
  let bytes = output_offset + mlp.output_bytes
  guard budget > 0L && bytes <= budget else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  { mlp, hidden: hidden_offset, output: output_offset, bytes, }
}

///|
/// Owns reusable F32 frequency, hidden and embedding storage. The final BF16
/// conditioning buffer is caller-owned and can be shared by every block.
pub struct TimestepProgram {
  priv source : @minimax_h3_block_source.TimestepSource
  priv functions : Array[@device.Function]
  priv mut workspace : @device.Allocation?
  priv mut layout : TimestepWorkspace?
}

///|
pub fn TimestepProgram::new(
  source : @minimax_h3_block_source.TimestepSource,
) -> Self {
  { source, functions: [], workspace: None, layout: None, }
}

///|
/// Retain this owner after failure and close acquired resources before retry.
pub fn TimestepProgram::prepare(
  self : Self,
  context : @device.Context,
  aot_module : @device.Module,
  max_workspace_bytes~ : Int64,
) -> Unit raise @device.DeviceError {
  guard self.functions.is_empty() && self.workspace is None else {
    raise @device.DeviceError::InvalidArgument(FunctionLoad)
  }
  let s = self.source
  let layout = timestep_workspace(
    s.rows,
    s.input_width,
    s.hidden_width,
    s.output_width,
    max_workspace_bytes,
  )
  acquire_functions(s.symbols[:], self.functions, name => {
    aot_module.load_function(@utf8.encode(name))
  })
  self.workspace = Some(context.allocate(layout.bytes))
  self.layout = Some(layout)
}

///|
pub fn TimestepProgram::close(self : Self) -> Unit raise @device.DeviceError {
  self.layout = None
  if self.workspace is Some(workspace) {
    workspace.close()
    self.workspace = None
  }
  release_functions(self.functions, function => function.close())
}

///|
/// Bind frequency -> dense/SiLU -> dense -> SiLU/BF16. All distinct times
/// are processed together; model block count does not multiply this work.
pub fn TimestepProgram::bind(
  self : Self,
  packing : @minimax_h3_weights.TimestepWeightPacking,
  timesteps~ : @device.Allocation,
  timestep_offset~ : Int64,
  weights~ : @device.Allocation,
  conditioning~ : @device.Allocation,
  conditioning_offset~ : Int64,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.layout is Some(layout) && self.workspace is Some(workspace) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let f = self.functions
  let launches = [
    layout.mlp.frequency_launch(
      f[0],
      timesteps~,
      timestep_offset~,
      output=workspace,
      output_offset=0L,
    ),
  ]
  launches.append(
    layout.mlp.launches(
      f[1],
      f[2],
      input=workspace,
      input_offset=0L,
      scratch=workspace,
      scratch_offset=layout.hidden,
      output=workspace,
      output_offset=layout.output,
      weights~,
      first_weight_offset=packing.first_weight_offset,
      first_bias_offset=packing.first_bias_offset,
      second_weight_offset=packing.second_weight_offset,
      second_bias_offset=packing.second_bias_offset,
    ),
  )
  launches.push(
    @joint_diffusion_device.conditioning_activation_launch(
      f[3],
      elements=self.source.rows.to_int64() * self.source.output_width.to_int64(),
      input=workspace,
      input_offset=layout.output,
      output=conditioning,
      output_offset=conditioning_offset,
    ),
  )
  launches
}
