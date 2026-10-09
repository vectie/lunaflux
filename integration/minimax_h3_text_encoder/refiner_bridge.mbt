///|
/// Bind completed layer-50 features into the existing context projection and
/// two-layer DiT refiner. Refiner owns a separate output; close its queue first.
pub fn EncoderRequest::prepare_refiner(
  self : Self,
  refiner : @minimax_h3_block_program.TextConditioningProgram,
  context : @device.Context,
  stream : @device.Stream,
  aot_module : @device.Module,
  layers : ArrayView[
    (@minimax_h3_weights.RefinerWeightPacking, @device.Allocation),
  ],
  weights~ : @device.Allocation,
  max_workspace_bytes~ : Int64,
  max_output_bytes~ : Int64,
) -> Unit raise @device.DeviceError {
  guard refiner.rows() == self.source.plan.rows else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  self.with_output((encoded, encoded_offset) => {
    refiner.prepare(
      context,
      stream,
      aot_module,
      layers,
      encoded~,
      encoded_offset~,
      weights~,
      max_workspace_bytes~,
      max_output_bytes~,
    )
  })
}

///|
/// Build the dependent refiner queue before either encoder or refiner runs.
/// The heterogeneous pipeline must execute encoder first, then this refiner.
pub fn EncoderRequest::prebind_refiner(
  self : Self,
  refiner : @minimax_h3_block_program.TextConditioningProgram,
  context : @device.Context,
  stream : @device.Stream,
  aot_module : @device.Module,
  layers : ArrayView[
    (@minimax_h3_weights.RefinerWeightPacking, @device.Allocation),
  ],
  weights~ : @device.Allocation,
  max_workspace_bytes~ : Int64,
  max_output_bytes~ : Int64,
) -> Unit raise @device.DeviceError {
  guard refiner.rows() == self.source.plan.rows else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  self.with_destination((encoded, encoded_offset) => {
    refiner.prepare(
      context,
      stream,
      aot_module,
      layers,
      encoded~,
      encoded_offset~,
      weights~,
      max_workspace_bytes~,
      max_output_bytes~,
    )
  })
}
