///|
/// One complete checkpoint prediction block: main-KV priming, or mHC-attention-
/// mHC plus the routed/shared FFN envelope. Both tables borrow parent completion;
/// main input and five draft residual rows remain separate caller-owned ports.
pub struct DeepSeekPredictionBlock {
  priv spec : @spec.DeepSeekV4ModelSpec
  priv address : @spec.DeepSeekV4BlockAddress
  priv hyper : @hyper.DeepSeekHyperConnection
  priv attention : DeepSeekPredictionAttention
  priv ffn : @expert.DeepSeekFfnEnvelope
  priv routed : @precision.ExpertMlpPrecision
  priv shared : @precision.ExpertMlpPrecision
  priv router : @upload.CheckpointDeviceWeights
  priv buffers : Array[@device.Allocation]
  priv residual_bytes : Int64
  priv main_rows : Int
  priv prime : Array[@device.OrderedKernelLaunch]
  priv predict : Array[@device.OrderedKernelLaunch]
  priv mut result : @device.Allocation?
  priv mut state : AttentionState
}

///|
pub fn DeepSeekPredictionBlock::new(
  spec : @spec.DeepSeekV4ModelSpec,
  stage~ : Int,
  maximum_main_rows~ : Int,
  maximum_positions~ : Int,
  maximum_workspace_bytes~ : Int64,
  maximum_expert_bytes~ : Int64,
  accuracy~ : @precision.Accuracy,
) -> Self raise {
  require_prediction_stage(spec, stage)
  let address = @spec.DeepSeekV4BlockAddress::PredictionStage(stage)
  let rows = spec.dspark_block_size()
  {
    spec,
    address,
    hyper: @hyper.DeepSeekHyperConnection::for_block(
      spec,
      block=address,
      branch=Attention,
      maximum_rows=rows,
    ),
    attention: DeepSeekPredictionAttention::new(
      spec,
      stage~,
      maximum_main_rows~,
      maximum_positions~,
      maximum_workspace_bytes~,
    ),
    ffn: @expert.DeepSeekFfnEnvelope::for_block(
      spec,
      block=address,
      maximum_rows=rows,
      maximum_expert_bytes~,
      accuracy~,
    ),
    routed: @expert.block_expert_compute_plan(
      spec,
      block=address,
      expert_indices=Array::makei(spec.routed_expert_count(), i => i),
      maximum_rows=rows,
      memory_ceiling=maximum_expert_bytes,
      accuracy~,
    ),
    shared: @expert.block_shared_expert_compute_plan(
      spec,
      block=address,
      maximum_rows=rows,
      memory_ceiling=maximum_expert_bytes,
      accuracy~,
    ),
    router: @upload.CheckpointDeviceWeights::new(),
    buffers: [],
    residual_bytes: rows.to_int64() *
    spec.hyper_connection_width().to_int64() *
    spec.hidden_size().to_int64() *
    2L,
    main_rows: maximum_main_rows,
    prime: [],
    predict: [],
    result: None,
    state: Fresh,
  }
}

///|
pub fn DeepSeekPredictionBlock::source_bytes(self : Self) -> Bytes raise {
  self.hyper.source_bytes() +
  b"\n" +
  self.attention.source_bytes() +
  b"\n" +
  self.ffn.source_bytes()
}

///|
pub fn DeepSeekPredictionBlock::bank_bytes(self : Self) -> Int64 {
  self.routed.bank_bytes() + self.shared.bank_bytes()
}

///|
/// Full compact bank, router, attention/mHC weights and workspace footprint.
/// Caller-owned residuals, counts/positions and normalized main input are not
/// counted again. Every checkpoint tensor is resolved before allocating banks.
pub fn DeepSeekPredictionBlock::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  ignore(
    @expert.bind_block_experts(
      checkpoint,
      self.spec,
      block=self.address,
      expert_indices=self.routed.local_experts(),
    ),
  )
  ignore(
    @expert.bind_block_shared_expert(checkpoint, self.spec, block=self.address),
  )
  @expert.bind_block_router(checkpoint, self.spec, block=self.address).fold(
    init=self.bank_bytes() +
      self.residual_bytes +
      self.hyper.required_device_bytes(checkpoint) +
      self.attention.required_device_bytes(checkpoint, output_borrowed=true) +
      self.ffn.required_device_bytes(checkpoint),
    (bytes, w) => bytes + w.buffer_bytes(),
  )
}

///|
pub fn DeepSeekPredictionBlock::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  main_counts~ : @device.Allocation,
  draft_counts~ : @device.Allocation,
  main_positions~ : @device.Allocation,
  draft_positions~ : @device.Allocation,
  main_hidden~ : @device.Allocation,
  input~ : @device.Allocation,
  output~ : @device.Allocation,
  reset_flag~ : @device.Allocation,
  start_position~ : @device.Allocation,
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming.StreamingSafetensorsLimits,
) -> Unit raise {
  guard self.state is Fresh &&
    !main_counts.same_storage(draft_counts) &&
    !input.same_storage(output) &&
    chunk_bytes > 0 else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  guard self.required_device_bytes(checkpoint) <= maximum_device_bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  for
    (b, bytes, alignment) in [
      (input, self.residual_bytes, 2L),
      (output, self.residual_bytes, 2L),
      (
        main_hidden,
        self.main_rows.to_int64() * self.spec.hidden_size().to_int64() * 2L,
        2L,
      ),
      (main_counts, 20L, 4L),
      (draft_counts, 20L, 4L),
      (main_positions, self.main_rows.to_int64() * 4L, 4L),
      (draft_positions, self.spec.dspark_block_size().to_int64() * 4L, 4L),
      (reset_flag, 4L, 4L),
      (start_position, 4L, 4L),
    ] {
    context.validate_allocation_region(
      b,
      offset=0L,
      byte_count=bytes,
      alignment~,
    )
  }
  self.state = Failed
  for
    bytes in [
      self.routed.bank_bytes(),
      self.shared.bank_bytes(),
      self.residual_bytes,
    ] {
    self.buffers.push(context.allocate(bytes))
  }
  @expert_frame.upload_bank(
    context,
    checkpoint,
    root,
    self.routed,
    self.buffers[0],
    weights=@expert.bind_block_experts(
      checkpoint,
      self.spec,
      block=self.address,
      expert_indices=self.routed.local_experts(),
    ),
    chunk_bytes~,
    limits~,
  )
  @expert_frame.upload_bank(
    context,
    checkpoint,
    root,
    self.shared,
    self.buffers[1],
    weights=@expert.bind_block_shared_expert(
      checkpoint,
      self.spec,
      block=self.address,
    ),
    chunk_bytes~,
    limits~,
  )
  let router = @expert.bind_block_router(
    checkpoint,
    self.spec,
    block=self.address,
  )
  self.router.upload(
    context,
    checkpoint,
    root,
    router,
    maximum_device_bytes=router.fold(init=0L, (bytes, w) => {
      bytes + w.buffer_bytes()
    }),
    chunk_bytes~,
    limits~,
  )
  self.hyper.prepare(
    context,
    aot,
    checkpoint,
    root,
    counts=draft_counts,
    input~,
    output=self.buffers[2],
    maximum_device_bytes=self.hyper.required_device_bytes(checkpoint),
    chunk_bytes~,
    limits~,
  )
  self.attention.prepare(
    context,
    aot,
    checkpoint,
    root,
    main_counts~,
    draft_counts~,
    main_positions~,
    draft_positions~,
    main_hidden~,
    draft_hidden=self.hyper.branch_input(),
    output=self.hyper.branch_output(),
    reset_flag~,
    start_position~,
    maximum_device_bytes=self.attention.required_device_bytes(
      checkpoint,
      output_borrowed=true,
    ),
    chunk_bytes~,
    limits~,
  )
  self.ffn.prepare(
    context,
    aot,
    checkpoint,
    root,
    counts=draft_counts,
    input=self.buffers[2],
    output~,
    routed_bank=self.buffers[0],
    shared_bank=self.buffers[1],
    router_weight=self.router.allocation(0),
    correction_bias=self.router.allocation(1),
    maximum_device_bytes=self.ffn.required_device_bytes(checkpoint),
    chunk_bytes~,
    limits~,
  )
  self.prime.append(self.attention.prime_launches())
  self.predict.append(self.hyper.prefix())
  self.predict.append(self.attention.prediction_launches())
  self.predict.append(self.hyper.suffix())
  self.predict.append(self.ffn.launches())
  self.result = Some(output)
  self.state = Prepared
}

///|
pub fn DeepSeekPredictionBlock::prime_launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.prime[:]
}

///|
pub fn DeepSeekPredictionBlock::prediction_launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.predict[:]
}

///|
pub fn DeepSeekPredictionBlock::output(self : Self) -> @device.Allocation raise {
  guard self.state is Prepared && self.result is Some(output) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  output
}

///|
pub fn DeepSeekPredictionBlock::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.prime.clear()
  self.predict.clear()
  self.result = None
  self.ffn.close()
  self.attention.close()
  self.hyper.close()
  self.router.close()
  while self.buffers.last() is Some(buffer) {
    buffer.close()
    ignore(self.buffers.pop())
  }
  self.state = Closed
}
