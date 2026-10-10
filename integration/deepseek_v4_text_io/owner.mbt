///|
priv enum State {
  Fresh
  Prepared
  Failed
  Closed
}

///|
/// Checkpoint-backed learned output boundary. Close the borrowing decoder queue
/// before this owner; both frame scratch and compact streamed weights are owned.
pub struct DeepSeekTextIo {
  priv spec : @spec.DeepSeekV4ModelSpec
  priv frame : @text.LearnedTextIoFrame
  priv weights : @upload.CheckpointDeviceWeights
  priv mut state : State
}

///|
pub fn DeepSeekTextIo::new(
  spec : @spec.DeepSeekV4ModelSpec,
  maximum_rows~ : Int,
  maximum_output_rows~ : Int,
  maximum_workspace_bytes~ : Int64,
) -> Self raise {
  {
    spec,
    frame: @text.LearnedTextIoFrame::new(
      learned_text_io_plan(
        spec,
        maximum_rows~,
        maximum_output_rows~,
        maximum_workspace_bytes~,
      ),
      symbol_prefix="deepseek_text_io",
    ),
    weights: @upload.CheckpointDeviceWeights::new(),
    state: Fresh,
  }
}

///|
pub fn DeepSeekTextIo::source_bytes(self : Self) -> Bytes {
  self.frame.source_bytes()
}

///|
pub fn DeepSeekTextIo::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  bind_text_io_weights(checkpoint, self.spec).fold(
    init=self.frame.workspace_bytes(),
    (bytes, weight) => bytes + weight.buffer_bytes(),
  )
}

///|
pub fn DeepSeekTextIo::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming.StreamingSafetensorsLimits,
  counts~ : @device.Allocation,
  output_count~ : @device.Allocation,
  selected_rows~ : @device.Allocation,
  output_residual~ : @device.Allocation,
  sampled_tokens~ : @device.Allocation,
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
    bind_text_io_weights(checkpoint, self.spec),
    maximum_device_bytes=maximum_device_bytes - self.frame.workspace_bytes(),
    chunk_bytes~,
    limits~,
  )
  self.frame.prepare(
    context,
    aot,
    counts~,
    output_count~,
    selected_rows~,
    output_residual~,
    norm_weight=self.weights.allocation(0),
    head_weight=self.weights.allocation(1),
    function_weight=self.weights.allocation(2),
    base=self.weights.allocation(3),
    scale=self.weights.allocation(4),
    sampled_tokens~,
    maximum_workspace_bytes=self.frame.workspace_bytes(),
  )
  self.state = Prepared
}

///|
pub fn DeepSeekTextIo::suffix(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.frame.suffix()
}

///|
pub fn DeepSeekTextIo::logits(self : Self) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.frame.logits()
}

///|
/// Borrow the exact prepared global vocabulary matrix, without re-uploading it.
/// Consumers must retire and close their queues before closing this owner.
pub fn DeepSeekTextIo::vocabulary_weight(
  self : Self,
) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.weights.allocation(1)
}

///|
pub fn DeepSeekTextIo::close(self : Self) -> Unit raise {
  match self.state {
    Closed => return
    Fresh | Prepared | Failed => ()
  }
  self.state = Failed
  self.frame.close()
  self.weights.close()
  self.state = Closed
}
