///|
/// Checkpoint-backed compression-zero attention through the hidden output.
/// The surrounding hyper-connection block owns normalization/residual mixing.
pub struct DeepSeekWindowSublayer {
  priv attention : DeepSeekWindowAttention
  priv output : DeepSeekAttentionOutput
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut state : State
}

///|
pub fn DeepSeekWindowSublayer::new(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  maximum_rows~ : Int,
  maximum_workspace_bytes~ : Int64,
) -> Self raise {
  let output_plan = attention_output_plan(
    spec,
    layer~,
    maximum_rows~,
    maximum_workspace_bytes~,
  )
  {
    attention: DeepSeekWindowAttention::new(
      spec,
      layer~,
      maximum_rows~,
      maximum_workspace_bytes=maximum_workspace_bytes -
        output_plan.workspace_bytes(),
    ),
    output: DeepSeekAttentionOutput::new(
      spec,
      layer~,
      maximum_rows~,
      maximum_workspace_bytes~,
    ),
    launches: [],
    state: Fresh,
  }
}

///|
pub fn DeepSeekWindowSublayer::source_bytes(self : Self) -> Bytes {
  self.attention.source_bytes() + b"\n" + self.output.source_bytes()
}

///|
pub fn DeepSeekWindowSublayer::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
  output_borrowed? : Bool = false,
) -> Int64 raise {
  self.attention.required_device_bytes(checkpoint) +
  self.output.required_device_bytes(checkpoint, output_borrowed~)
}

///|
pub fn DeepSeekWindowSublayer::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  counts~ : @device.Allocation,
  positions~ : @device.Allocation,
  reset_flag~ : @device.Allocation,
  hidden~ : @device.Allocation,
  output? : @device.Allocation,
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming.StreamingSafetensorsLimits,
) -> Unit raise {
  guard self.state is Fresh &&
    self.required_device_bytes(checkpoint, output_borrowed=output is Some(_)) <=
    maximum_device_bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  self.state = Failed
  self.attention.prepare(
    context,
    aot,
    checkpoint,
    root,
    counts~,
    positions~,
    reset_flag~,
    hidden~,
    maximum_device_bytes=maximum_device_bytes -
      self.output.required_device_bytes(
        checkpoint,
        output_borrowed=output is Some(_),
      ),
    chunk_bytes~,
    limits~,
  )
  self.output.prepare(
    context,
    aot,
    checkpoint,
    root,
    counts~,
    positions~,
    attention=self.attention.output(),
    output?,
    maximum_device_bytes=self.output.required_device_bytes(
      checkpoint,
      output_borrowed=output is Some(_),
    ),
    chunk_bytes~,
    limits~,
  )
  self.launches.append(self.attention.launches())
  self.launches.append(self.output.launches())
  self.state = Prepared
}

///|
pub fn DeepSeekWindowSublayer::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
pub fn DeepSeekWindowSublayer::output(self : Self) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.output.output()
}

///|
pub fn DeepSeekWindowSublayer::append_descriptor(
  self : Self,
) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.attention.append_descriptor()
}

///|
pub fn DeepSeekWindowSublayer::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.launches.clear()
  self.output.close()
  self.attention.close()
  self.state = Closed
}
