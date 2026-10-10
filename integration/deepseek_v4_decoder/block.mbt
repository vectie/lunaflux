///|
priv enum State {
  Fresh
  Prepared
  Failed
  Closed
}

///|
/// Complete single-rank base decoder block: attention mHC -> attention ->
/// residual publication -> FFN mHC -> routed/shared MoE -> residual publication.
/// The containing stage owns compact expert/router banks and the only queue.
pub struct DeepSeekDecoderBlock {
  priv attention : @attention.DeepSeekAttentionEnvelope
  priv ffn : @expert.DeepSeekFfnEnvelope
  priv residual_bytes : Int64
  priv mut intermediate : @device.Allocation?
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut state : State
}

///|
pub fn DeepSeekDecoderBlock::new(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  maximum_rows~ : Int,
  maximum_context~ : Int,
  maximum_state_bytes~ : Int64,
  maximum_workspace_bytes~ : Int64,
  maximum_expert_bytes~ : Int64,
  accuracy~ : @precision.Accuracy,
) -> Self raise {
  {
    attention: @attention.DeepSeekAttentionEnvelope::new(
      spec,
      layer~,
      maximum_rows~,
      maximum_context~,
      maximum_state_bytes~,
      maximum_workspace_bytes~,
    ),
    ffn: @expert.DeepSeekFfnEnvelope::new(
      spec,
      layer~,
      maximum_rows~,
      maximum_expert_bytes~,
      accuracy~,
    ),
    residual_bytes: maximum_rows.to_int64() *
    spec.hidden_size().to_int64() *
    spec.hyper_connection_width().to_int64() *
    2L,
    intermediate: None,
    launches: [],
    state: Fresh,
  }
}

///|
pub fn DeepSeekDecoderBlock::source_bytes(self : Self) -> Bytes raise {
  self.attention.source_bytes() + b"\n" + self.ffn.source_bytes()
}

///|
pub fn DeepSeekDecoderBlock::bank_bytes(self : Self) -> (Int64, Int64) {
  self.ffn.bank_bytes()
}

///|
/// Private frames and streamed non-expert weights only. Add caller-owned
/// expert/router/token-table banks and residual ports once in the stage budget.
pub fn DeepSeekDecoderBlock::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  self.residual_bytes +
  self.attention.required_device_bytes(checkpoint) +
  self.ffn.required_device_bytes(checkpoint)
}

///|
pub fn DeepSeekDecoderBlock::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  counts~ : @device.Allocation,
  positions~ : @device.Allocation,
  reset_flag~ : @device.Allocation,
  input~ : @device.Allocation,
  output~ : @device.Allocation,
  index_offsets? : @device.Allocation,
  routed_bank~ : @device.Allocation,
  shared_bank~ : @device.Allocation,
  router_weight~ : @device.Allocation,
  correction_bias? : @device.Allocation,
  token_ids? : @device.Allocation,
  token_table? : @device.Allocation,
  token_table_offset? : Int64,
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming.StreamingSafetensorsLimits,
) -> Unit raise {
  guard self.state is Fresh &&
    self.required_device_bytes(checkpoint) <= maximum_device_bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  context.validate_allocation_region(
    input,
    offset=0L,
    byte_count=self.residual_bytes,
    alignment=2L,
  )
  context.validate_allocation_region(
    output,
    offset=0L,
    byte_count=self.residual_bytes,
    alignment=2L,
  )
  self.state = Failed
  let intermediate = context.allocate(self.residual_bytes)
  self.intermediate = Some(intermediate)
  self.attention.prepare(
    context,
    aot,
    checkpoint,
    root,
    counts~,
    positions~,
    reset_flag~,
    input~,
    output=intermediate,
    index_offsets?,
    maximum_device_bytes=self.attention.required_device_bytes(checkpoint),
    chunk_bytes~,
    limits~,
  )
  self.ffn.prepare(
    context,
    aot,
    checkpoint,
    root,
    counts~,
    input=intermediate,
    output~,
    routed_bank~,
    shared_bank~,
    router_weight~,
    correction_bias?,
    token_ids?,
    token_table?,
    token_table_offset?,
    maximum_device_bytes=self.ffn.required_device_bytes(checkpoint),
    chunk_bytes~,
    limits~,
  )
  self.launches.append(self.attention.launches())
  self.launches.append(self.ffn.launches())
  self.state = Prepared
}

///|
pub fn DeepSeekDecoderBlock::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
pub fn DeepSeekDecoderBlock::append_descriptors(
  self : Self,
) -> ArrayView[@device.Allocation] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.attention.append_descriptors()
}

///|
pub fn DeepSeekDecoderBlock::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.launches.clear()
  self.ffn.close()
  self.attention.close()
  if self.intermediate is Some(buffer) {
    buffer.close()
    self.intermediate = None
  }
  self.state = Closed
}
