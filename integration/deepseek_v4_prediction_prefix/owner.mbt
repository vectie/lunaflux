///|
priv enum State {
  Fresh
  Failed
  Prepared
  Closed
}

///|
/// Executable main-hidden prefix, not a complete DSpark predictor. Input must
/// contain every official target in order, not a rank-local partial capture.
pub struct DeepSeekPredictionMain {
  priv spec : @spec.DeepSeekV4ModelSpec
  priv frame : @frame.RowProjectionNormFrame
  priv weights : @upload.CheckpointDeviceWeights
  priv mut state : State
}

///|
pub fn DeepSeekPredictionMain::new(
  spec : @spec.DeepSeekV4ModelSpec,
  maximum_rows~ : Int,
  maximum_workspace_bytes~ : Int64,
) -> Self raise {
  {
    spec,
    frame: @frame.RowProjectionNormFrame::new(
      @source.row_projection_norm_source(
        prediction_main_plan(spec, maximum_rows~, maximum_workspace_bytes~),
        symbol_prefix="deepseek_prediction_main",
      ),
    ),
    weights: @upload.CheckpointDeviceWeights::new(),
    state: Fresh,
  }
}

///|
pub fn DeepSeekPredictionMain::source_bytes(self : Self) -> Bytes {
  self.frame.source_bytes()
}

///|
pub fn DeepSeekPredictionMain::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  bind_prediction_main_weights(checkpoint, self.spec).fold(
    init=self.frame.plan().workspace_bytes(),
    (total, w) => total + w.buffer_bytes(),
  )
}

///|
pub fn DeepSeekPredictionMain::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  counts~ : @device.Allocation,
  captured_hidden~ : @device.Allocation,
  target_layers~ : ArrayView[Int],
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming.StreamingSafetensorsLimits,
) -> Unit raise {
  guard self.state is Fresh &&
    target_layers == self.spec.dspark_target_layer_ids()[:] else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  guard self.required_device_bytes(checkpoint) <= maximum_device_bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  context.validate_allocation_region(
    counts,
    offset=0L,
    byte_count=20L,
    alignment=4L,
  )
  context.validate_allocation_region(
    captured_hidden,
    offset=0L,
    byte_count=self.frame.plan().input_bytes(),
    alignment=2L,
  )
  self.state = Failed
  self.weights.upload(
    context,
    checkpoint,
    root,
    bind_prediction_main_weights(checkpoint, self.spec),
    maximum_device_bytes=maximum_device_bytes -
      self.frame.plan().workspace_bytes(),
    chunk_bytes~,
    limits~,
  )
  self.frame.prepare(
    context,
    aot,
    counts~,
    input=captured_hidden,
    matrix=self.weights.allocation(0),
    norm=self.weights.allocation(1),
    maximum_workspace_bytes=self.frame.plan().workspace_bytes(),
  )
  self.state = Prepared
}

///|
pub fn DeepSeekPredictionMain::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.frame.launches()
}

///|
pub fn DeepSeekPredictionMain::output(self : Self) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.frame.output()
}

///|
pub fn DeepSeekPredictionMain::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.frame.close()
  self.weights.close()
  self.state = Closed
}
