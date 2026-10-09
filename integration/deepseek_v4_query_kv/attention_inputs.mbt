///|
/// Checkpoint-backed prefix through rotary and simulated cache precision.
/// This publishes attention operands, not a cache commit or attention result.
pub struct DeepSeekAttentionInputs {
  priv prefix : DeepSeekQueryKv
  priv rotary : @rotary_frame.RotaryKvFrame
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut state : State
}

///|
pub(all) struct DeepSeekAttentionInputBuffers {
  low_rank_queries : @device.Allocation
  queries : @device.Allocation
  shared_kv : @device.Allocation
}

///|
pub fn DeepSeekAttentionInputs::new(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  maximum_rows~ : Int,
  maximum_workspace_bytes~ : Int64,
) -> Self raise {
  let plan = rotary_kv_plan(
    spec,
    layer~,
    maximum_rows~,
    maximum_workspace_bytes~,
  )
  {
    prefix: DeepSeekQueryKv::new(
      spec,
      layer~,
      maximum_rows~,
      maximum_workspace_bytes=maximum_workspace_bytes - plan.workspace_bytes(),
    ),
    rotary: @rotary_frame.RotaryKvFrame::new(
      @rotary_source.rotary_kv_source(
        plan,
        symbol_prefix="deepseek_l\{layer}_attention_inputs",
      ),
    ),
    launches: [],
    state: Fresh,
  }
}

///|
pub fn DeepSeekAttentionInputs::source_bytes(self : Self) -> Bytes {
  self.prefix.source_bytes() + b"\n" + self.rotary.source_bytes()
}

///|
pub fn DeepSeekAttentionInputs::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  self.prefix.required_device_bytes(checkpoint) + self.rotary.workspace_bytes()
}

///|
pub fn DeepSeekAttentionInputs::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  counts~ : @device.Allocation,
  positions~ : @device.Allocation,
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
  self.prefix.prepare(
    context,
    aot,
    checkpoint,
    root,
    counts~,
    hidden~,
    chunk_bytes~,
    limits~,
    maximum_device_bytes=maximum_device_bytes - self.rotary.workspace_bytes(),
  )
  let input = self.prefix.outputs()
  self.rotary.prepare(
    context,
    aot,
    counts~,
    positions~,
    queries=input.queries,
    shared_kv=input.shared_kv,
    maximum_workspace_bytes=self.rotary.workspace_bytes(),
  )
  self.launches.append(self.prefix.launches())
  self.launches.append(self.rotary.launches())
  self.state = Prepared
}

///|
/// All eight operations enter the caller's same queue and completion.
pub fn DeepSeekAttentionInputs::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
pub fn DeepSeekAttentionInputs::outputs(
  self : Self,
) -> DeepSeekAttentionInputBuffers raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let prefix = self.prefix.outputs()
  let rotary = self.rotary.outputs()
  {
    low_rank_queries: prefix.low_rank_queries,
    queries: rotary.queries,
    shared_kv: rotary.shared_kv,
  }
}

///|
pub fn DeepSeekAttentionInputs::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.launches.clear()
  self.rotary.close()
  self.prefix.close()
  self.state = Closed
}
