///|
priv enum FfnState {
  Fresh
  Prepared
  Failed
  Closed
}

///|
/// Complete single-rank FFN residual envelope. Compact expert banks and router
/// operands belong to the stage loader. This owner never duplicates those
/// banks or introduces another execution/completion queue.
pub struct DeepSeekFfnEnvelope {
  priv hyper : @hyper.DeepSeekHyperConnection
  priv program : @expert.MoeProgram
  priv routed_bytes : Int64
  priv shared_bytes : Int64
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut state : FfnState
}

///|
pub fn DeepSeekFfnEnvelope::new(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  maximum_rows~ : Int,
  maximum_expert_bytes~ : Int64,
  accuracy~ : @precision.Accuracy,
) -> Self raise {
  let routed = expert_compute_plan(
    spec,
    layer_index=layer,
    expert_indices=Array::makei(spec.routed_expert_count(), i => i),
    maximum_rows~,
    memory_ceiling=maximum_expert_bytes,
    accuracy~,
  )
  let shared = shared_expert_compute_plan(
    spec,
    layer_index=layer,
    maximum_rows~,
    memory_ceiling=maximum_expert_bytes,
    accuracy~,
  )
  guard routed.bank_bytes() <= maximum_expert_bytes &&
    shared.bank_bytes() <= maximum_expert_bytes - routed.bank_bytes() else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  let program = @expert.MoeProgram::new(
    @expert_source.packed_expert_mlp_source(
      routed,
      symbol="deepseek_l\{layer}_routed",
    ),
    @expert_source.packed_expert_mlp_source(
      shared,
      symbol="deepseek_l\{layer}_shared",
    ),
    @combine_source.moe_combine_source(
      @precision.MoeCombinePrecision::new(
        maximum_rows~,
        hidden=spec.hidden_size(),
        rounding=SumThenBf16,
        memory_ceiling=maximum_rows.to_int64() *
          spec.hidden_size().to_int64() *
          10L,
      ),
      symbol="deepseek_l\{layer}_moe_combine",
    ),
    routing=@router_source.routing_source(
      routing_compute_plan(
        spec,
        layer_index=layer,
        maximum_rows~,
        maximum_workspace_bytes=maximum_expert_bytes,
        accuracy~,
      ),
      symbol="deepseek_l\{layer}_router",
    ),
  )
  {
    hyper: @hyper.DeepSeekHyperConnection::new(
      spec,
      layer~,
      branch=FeedForward,
      maximum_rows~,
    ),
    program,
    routed_bytes: routed.bank_bytes(),
    shared_bytes: shared.bank_bytes(),
    launches: [],
    state: Fresh,
  }
}

///|
pub fn DeepSeekFfnEnvelope::source_bytes(self : Self) -> Bytes raise {
  self.hyper.source_bytes() + b"\n" + self.program.source_bytes()
}

///|
/// Caller-owned compact bank capacities, not included in private frame bytes.
pub fn DeepSeekFfnEnvelope::bank_bytes(self : Self) -> (Int64, Int64) {
  (self.routed_bytes, self.shared_bytes)
}

///|
pub fn DeepSeekFfnEnvelope::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  self.hyper.required_device_bytes(checkpoint) + self.program.workspace_bytes()
}

///|
pub fn DeepSeekFfnEnvelope::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  counts~ : @device.Allocation,
  input~ : @device.Allocation,
  output~ : @device.Allocation,
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
    routed_bank,
    offset=0L,
    byte_count=self.routed_bytes,
    alignment=1L,
  )
  context.validate_allocation_region(
    shared_bank,
    offset=0L,
    byte_count=self.shared_bytes,
    alignment=1L,
  )
  self.state = Failed
  self.hyper.prepare(
    context,
    aot,
    checkpoint,
    root,
    counts~,
    input~,
    output~,
    maximum_device_bytes=maximum_device_bytes - self.program.workspace_bytes(),
    chunk_bytes~,
    limits~,
  )
  self.program.prepare_routed_frame(
    context,
    aot,
    maximum_workspace_bytes=self.program.workspace_bytes(),
    counts~,
    input=self.hyper.branch_input(),
    output=self.hyper.branch_output(),
    routed_bank~,
    shared_bank~,
    router_weight~,
    correction_bias?,
    token_ids?,
    token_table?,
    token_table_offset?,
  )
  self.launches.append(self.hyper.prefix())
  self.launches.append(self.program.launches())
  self.launches.append(self.hyper.suffix())
  self.state = Prepared
}

///|
pub fn DeepSeekFfnEnvelope::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
pub fn DeepSeekFfnEnvelope::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.launches.clear()
  self.program.close()
  self.hyper.close()
  self.state = Closed
}
