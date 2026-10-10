///|
priv enum State {
  Fresh
  Prepared
  Failed
  Closed
}

///|
/// Owns checkpoint control weights and prepared effects. The decoder borrows
/// prefix/suffix and writes branch_output between them in its own queue.
pub struct DeepSeekHyperConnection {
  priv spec : @spec.DeepSeekV4ModelSpec
  priv block : @spec.DeepSeekV4BlockAddress
  priv branch : DeepSeekHyperBranch
  priv frame : @frame.HyperConnectionFrame
  priv weights : @upload.CheckpointDeviceWeights
  priv mut state : State
}

///|
pub fn DeepSeekHyperConnection::new(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  branch~ : DeepSeekHyperBranch,
  maximum_rows~ : Int,
) -> Self raise {
  guard layer >= 0 && layer < spec.layer_count() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  DeepSeekHyperConnection::for_block(
    spec,
    block=BaseLayer(layer),
    branch~,
    maximum_rows~,
  )
}

///|
pub fn DeepSeekHyperConnection::for_block(
  spec : @spec.DeepSeekV4ModelSpec,
  block~ : @spec.DeepSeekV4BlockAddress,
  branch~ : DeepSeekHyperBranch,
  maximum_rows~ : Int,
) -> Self raise {
  let prefix = block.symbol_prefix(spec)
  let name = match branch {
    Attention => "attn"
    FeedForward => "ffn"
  }
  {
    spec,
    block,
    branch,
    frame: @frame.HyperConnectionFrame::new(
      @source.hyper_connection_source(
        hyper_connection_plan(spec, maximum_rows~),
        symbol_prefix=prefix + "_hyper_" + name,
      ),
    ),
    weights: @upload.CheckpointDeviceWeights::new(),
    state: Fresh,
  }
}

///|
pub fn DeepSeekHyperConnection::source_bytes(self : Self) -> Bytes {
  self.frame.source_bytes()
}

///|
pub fn DeepSeekHyperConnection::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  bind_block_hyper_connection_weights(
    checkpoint,
    self.spec,
    block=self.block,
    branch=self.branch,
  ).fold(init=self.frame.workspace_bytes(), (total, weight) => {
    total + weight.buffer_bytes()
  })
}

///|
pub fn DeepSeekHyperConnection::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  counts~ : @device.Allocation,
  input~ : @device.Allocation,
  output~ : @device.Allocation,
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
    bind_block_hyper_connection_weights(
      checkpoint,
      self.spec,
      block=self.block,
      branch=self.branch,
    ),
    maximum_device_bytes=maximum_device_bytes - self.frame.workspace_bytes(),
    chunk_bytes~,
    limits~,
  )
  self.frame.prepare(
    context,
    aot,
    counts~,
    input~,
    output~,
    function_weight=self.weights.allocation(0),
    base=self.weights.allocation(1),
    scale=self.weights.allocation(2),
    norm_weight=self.weights.allocation(3),
    maximum_workspace_bytes=self.frame.workspace_bytes(),
  )
  self.state = Prepared
}

///|
pub fn DeepSeekHyperConnection::prefix(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.frame.prefix()
}

///|
pub fn DeepSeekHyperConnection::suffix(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.frame.suffix()
}

///|
pub fn DeepSeekHyperConnection::branch_input(
  self : Self,
) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.frame.branch_input()
}

///|
pub fn DeepSeekHyperConnection::branch_output(
  self : Self,
) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.frame.branch_output()
}

///|
pub fn DeepSeekHyperConnection::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.frame.close()
  self.weights.close()
  self.state = Closed
}
