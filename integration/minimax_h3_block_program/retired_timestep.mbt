///|
/// Offline observer after the prediction prefix retires. The original DiT
/// consumes this F32 embedding before SiLU/BF16 parameter projection.
pub fn[T] PredictionVariant::with_timestep_embedding(
  self : Self,
  observe : (@device.Allocation, Int64, Int64) -> T raise,
) -> T raise {
  guard self.timestep.workspace is Some(workspace) &&
    self.timestep.layout is Some(layout) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  observe(workspace, layout.output, layout.mlp.output_bytes)
}

///|
/// Prepared producer extent, not a guessed index into the full denoiser list.
pub fn PredictionVariant::input_launch_count(
  self : Self,
  input : InputProgram,
) -> Int {
  self.timestep.source.symbols.length() + input.source.symbols.length()
}
