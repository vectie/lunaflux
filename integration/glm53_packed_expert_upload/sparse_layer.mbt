///|
/// Full checkpoint-backed Flash DSA block: both mHC envelopes, projected
/// retained sparse attention, output projection, router and routed/shared FFN.
/// One request owns one history; this is not multi-request cache scheduling.
pub struct GlmSparseLayer {
  priv spec : @glm53_spec.Glm53FlashSpec
  priv layer_index : Int
  priv maximum_rows : Int
  priv decoder : @sparse_frame.IndexedSparseDecoderFrame
  priv ffn : GlmFfn
  priv weights : @upload.CheckpointDeviceWeights
  priv mut queue : @device.OrderedKernelExecutor?
  priv mut launch_count : Int
  priv mut status : RecurrentLayerState
}

///|
pub fn GlmSparseLayer::new(
  spec : @glm53_spec.Glm53FlashSpec,
  layer_index~ : Int,
  maximum_queries~ : Int,
  maximum_history~ : Int,
  index_layer_norm_epsilon~ : Double,
  maximum_state_bytes~ : Int64,
  maximum_workspace_bytes~ : Int64,
  maximum_expert_bytes~ : Int64,
  accuracy~ : @precision_ir.Accuracy,
) -> Self raise {
  let projection = @sparse_model.flash_sparse_projection_plan(
    spec,
    layer_index~,
    maximum_queries~,
    maximum_history~,
    index_layer_norm_epsilon~,
    maximum_attention_workspace_bytes=maximum_workspace_bytes,
    maximum_projection_workspace_bytes=maximum_workspace_bytes,
  )
  let envelope = @precision_ir.HyperConnectionPrecision::new(
    stream_count=spec.hyper_connection_streams(),
    hidden_width=spec.hidden_size(),
    maximum_rows=maximum_queries,
    sinkhorn_iterations=spec.hyper_connection_sinkhorn_iterations(),
    control_epsilon=spec.hyper_connection_epsilon(),
    normalization_epsilon=spec.rms_norm_epsilon(),
    mix_law=RoundedBf16ControlsAndProducts,
  )
  let plan = @precision_ir.IndexedSparseDecoderPrecision::new(
    projection,
    envelope,
    maximum_state_bytes~,
    maximum_workspace_bytes~,
  )
  let ffn = make_moe_ffn(
    spec, layer_index, maximum_queries, maximum_expert_bytes, accuracy,
  )
  guard ffn.workspace_bytes() <=
    maximum_workspace_bytes - plan.workspace_bytes() else {
    raise @precision_ir.PrecisionPlanError::MemoryBudgetExceeded
  }
  {
    spec,
    layer_index,
    maximum_rows: maximum_queries,
    decoder: @sparse_frame.IndexedSparseDecoderFrame::new(
      @sparse_source.indexed_sparse_decoder_source(
        plan,
        symbol_prefix="glm_sparse_" + layer_index.to_string(),
      ),
    ),
    ffn,
    weights: @upload.CheckpointDeviceWeights::new(),
    queue: None,
    launch_count: 0,
    status: Fresh,
  }
}

///|
pub fn GlmSparseLayer::source_bytes(self : Self) -> Bytes {
  self.decoder.source_bytes() + b"\n" + self.ffn.source_bytes()
}

///|
pub fn GlmSparseLayer::workspace_bytes(self : Self) -> Int64 {
  self.decoder.workspace_bytes() + self.ffn.workspace_bytes()
}

///|
pub fn GlmSparseLayer::state_bytes(self : Self) -> Int64 {
  self.decoder.state_bytes()
}

///|
pub fn GlmSparseLayer::device_bytes(self : Self) -> Int64 {
  self.workspace_bytes() +
  self.state_bytes() +
  self.weights.device_bytes() +
  self.ffn.owned_bank_bytes()
}

///|
pub fn GlmSparseLayer::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  stream : @device.Stream,
  checkpoint : @precision_materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming_safetensors.StreamingSafetensorsLimits,
  counts~ : @device.Allocation,
  reset_flag~ : @device.Allocation,
  hidden~ : @device.Allocation,
  output~ : @device.Allocation,
) -> Unit raise {
  let frames = self.workspace_bytes() + self.state_bytes()
  let banks = self.ffn.bank_bytes()
  guard self.status is Fresh &&
    frames <= maximum_device_bytes &&
    banks <= maximum_device_bytes - frames else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.status = Failed
  let bound = @sparse_model.bind_flash_sparse_projection_weights(
    checkpoint,
    self.spec,
    layer_index=self.layer_index,
  )
  bound.append(
    @sparse_model.bind_flash_sparse_output_weights(
      checkpoint,
      self.spec,
      layer_index=self.layer_index,
    ),
  )
  bound.append(
    bind_attention_hyper_weights(checkpoint, self.spec, self.layer_index),
  )
  let ffn_offset = bound.length()
  bound.append(bind_router(checkpoint, layer_index=self.layer_index))
  bound.append(bind_ffn_hyper_weights(checkpoint, self.spec, self.layer_index))
  self.weights.upload(
    context,
    checkpoint,
    root,
    bound,
    maximum_device_bytes=maximum_device_bytes - frames - banks,
    chunk_bytes~,
    limits~,
  )
  self.ffn.prepare(
    context,
    aot,
    self.weights,
    checkpoint,
    root,
    layer_index=self.layer_index,
    chunk_bytes~,
    limits~,
    counts~,
    output~,
    weight_offset=ffn_offset,
  )
  self.decoder.prepare(
    context,
    aot,
    maximum_workspace_bytes=self.decoder.workspace_bytes(),
    maximum_state_bytes=self.decoder.state_bytes(),
    counts~,
    reset_flag~,
    hidden~,
    output=self.ffn.input(),
    projection_weights=Array::makei(12, i => self.weights.allocation(i)),
    output_weight=self.weights.allocation(12),
    pool_ape=self.weights.allocation(13),
    hyper_weights=Array::makei(4, i => self.weights.allocation(14 + i)),
  )
  let launches = self.decoder.launches().to_owned()
  launches.append(self.ffn.launches)
  self.queue = Some(context.create_ordered_kernel_executor(stream, launches))
  self.launch_count = launches.length()
  self.status = Ready
}

///|
/// Borrowed overflow port for output-delivery error propagation.
pub fn GlmSparseLayer::append_descriptor(
  self : Self,
) -> @device.Allocation raise @device.DeviceError {
  self.decoder.append_descriptor()
}

///|
pub fn GlmSparseLayer::submit(
  self : Self,
  live_rows~ : Int,
) -> Unit raise @device.DeviceError {
  guard self.status is Ready && self.queue is Some(queue) else {
    raise @device.DeviceError::Busy(OrderedExecutorEnqueue)
  }
  guard live_rows > 0 && live_rows <= self.maximum_rows else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  self.status = Failed
  for index in 0..<self.launch_count {
    queue.enqueue(index)
  }
  queue.record_completion()
  self.status = Active
}

///|
pub fn GlmSparseLayer::poll(self : Self) -> Bool raise @device.DeviceError {
  guard self.status is Active && self.queue is Some(queue) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  self.status = Failed
  if queue.poll_completion() {
    queue.reset()
    self.status = Ready
    true
  } else {
    self.status = Active
    false
  }
}

///|
pub fn GlmSparseLayer::close(self : Self) -> Unit raise @device.DeviceError {
  if self.status is Closed {
    return
  }
  self.status = Failed
  if self.queue is Some(queue) {
    queue.abort()
    queue.close()
    self.queue = None
  }
  self.ffn.close()
  self.decoder.close()
  self.weights.close()
  self.status = Closed
}
