///|
priv enum State {
  Fresh
  Prepared
  Failed
  Closed
}

///|
/// Checkpoint-backed query/KV execution prefix. Rotary, simulated KV
/// quantization, compression and cache ownership are deliberately downstream.
pub struct DeepSeekQueryKv {
  priv spec : @spec.DeepSeekV4ModelSpec
  priv layer : Int
  priv frame : @frame.QuantizedQueryKvFrame
  priv weights : @upload.CheckpointDeviceWeights
  priv mut state : State
}

///|
pub fn DeepSeekQueryKv::new(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  maximum_rows~ : Int,
  maximum_workspace_bytes~ : Int64,
) -> Self raise {
  guard layer >= 0 && layer < spec.layer_count() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  {
    spec,
    layer,
    frame: @frame.QuantizedQueryKvFrame::new(
      @source.quantized_query_kv_source(
        query_kv_plan(spec, maximum_rows~, maximum_workspace_bytes~),
        symbol_prefix="deepseek_l\{layer}_query_kv",
      ),
    ),
    weights: @upload.CheckpointDeviceWeights::new(),
    state: Fresh,
  }
}

///|
pub fn DeepSeekQueryKv::source_bytes(self : Self) -> Bytes {
  self.frame.source_bytes()
}

///|
pub fn DeepSeekQueryKv::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  bind_query_kv_weights(checkpoint, self.spec, self.layer).fold(
    init=self.frame.workspace_bytes(),
    (bytes, weight) => bytes + weight.buffer_bytes(),
  )
}

///|
pub fn DeepSeekQueryKv::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  counts~ : @device.Allocation,
  hidden~ : @device.Allocation,
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming.StreamingSafetensorsLimits,
) -> Unit raise {
  guard self.state is Fresh &&
    self.required_device_bytes(checkpoint) <= maximum_device_bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  self.state = Failed
  self.weights.upload(
    context,
    checkpoint,
    root,
    bind_query_kv_weights(checkpoint, self.spec, self.layer),
    maximum_device_bytes=maximum_device_bytes - self.frame.workspace_bytes(),
    chunk_bytes~,
    limits~,
  )
  self.frame.prepare(
    context,
    aot,
    counts~,
    hidden~,
    weights=[
      self.weights.allocation(0),
      self.weights.allocation(1),
      self.weights.allocation(2),
      self.weights.allocation(3),
      self.weights.allocation(4),
    ],
    maximum_workspace_bytes=self.frame.workspace_bytes(),
  )
  self.state = Prepared
}

///|
pub fn DeepSeekQueryKv::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.frame.launches()
}

///|
pub fn DeepSeekQueryKv::outputs(
  self : Self,
) -> @frame.QueryKvProjectionOutputs raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.frame.outputs()
}

///|
/// Borrowing queue first, then frame/intermediates, then checkpoint weights.
pub fn DeepSeekQueryKv::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.frame.close()
  self.weights.close()
  self.state = Closed
}
