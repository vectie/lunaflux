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
  priv tensor_group : (Int, Int)?
  priv weights : @upload.CheckpointDeviceWeights
  priv mut queue : @device.OrderedKernelExecutor?
  priv mut launch_count : Int
  priv launches : Array[@device.OrderedKernelLaunch]
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
  hc_reduction? : @precision_ir.HyperControlReduction = Ordered,
  matrix_law? : @precision_ir.DenseDotLaw = ScalarFloat32,
  matrix_staging? : @matrix.MatrixStaging = Synchronous,
  expert_scaled_dot_law? : @precision_ir.ScaledDotReductionLaw = DecodeOperandsFloat32,
  expert_transport? : @matrix.ExpertTransport = IndexedRows,
  shared_matrix_columns? : Int = 64,
  expert_fragment_slots? : Int = 1,
  expert_scale_partitions? : Int = 1,
  tensor_group? : (Int, Int),
  paged_history? : @history_domain.PagedHistoryDomain,
  attention_accumulation? : @precision_ir.AttentionAccumulationLaw = OrderedFloat32,
  index_scoring? : @precision_ir.IndexScoringLaw = OrderedBFloat16,
  sparse_cache? : @precision_ir.SparseCacheLaw = ExpandedBFloat16,
) -> Self raise {
  let projection = @sparse_model.flash_sparse_projection_plan(
    spec,
    layer_index~,
    maximum_queries~,
    maximum_history~,
    index_layer_norm_epsilon~,
    maximum_attention_workspace_bytes=maximum_workspace_bytes,
    maximum_projection_workspace_bytes=maximum_workspace_bytes,
    projection_law=matrix_law,
    paged_history?,
    attention_accumulation~,
    index_scoring~,
    cache_law=sparse_cache,
    tensor_group?,
  )
  let envelope = @precision_ir.HyperConnectionPrecision::new(
    stream_count=spec.hyper_connection_streams(),
    hidden_width=spec.hidden_size(),
    maximum_rows=maximum_queries,
    sinkhorn_iterations=spec.hyper_connection_sinkhorn_iterations(),
    control_epsilon=spec.hyper_connection_epsilon(),
    normalization_epsilon=spec.rms_norm_epsilon(),
    mix_law=RoundedBf16ControlsAndProducts,
    control_square_law=hc_reduction.square_law(),
    control_dot_law=hc_reduction.dot_law(),
  )
  let plan = @precision_ir.IndexedSparseDecoderPrecision::new(
    projection,
    envelope,
    maximum_state_bytes~,
    maximum_workspace_bytes~,
    publication=if tensor_group is Some(_) {
      ReduceFloat32BeforeRound
    } else {
      LocalBFloat16
    },
  )
  let ffn = make_moe_ffn(
    spec,
    layer_index,
    maximum_queries,
    maximum_expert_bytes,
    accuracy,
    hc_reduction~,
    matrix_law~,
    expert_scaled_dot_law~,
    expert_transport~,
    shared_matrix_columns~,
    expert_fragment_slots~,
    expert_scale_partitions~,
    tensor_group?,
  )
  let decoder = @sparse_frame.IndexedSparseDecoderFrame::new(
    @sparse_source.indexed_sparse_decoder_source(
      plan,
      symbol_prefix="glm_sparse_" + layer_index.to_string(),
      staging=matrix_staging,
    ),
  )
  guard ffn.workspace_bytes() <=
    maximum_workspace_bytes - decoder.workspace_bytes() else {
    raise @precision_ir.PrecisionPlanError::MemoryBudgetExceeded
  }
  {
    spec,
    layer_index,
    maximum_rows: maximum_queries,
    decoder,
    ffn,
    tensor_group,
    weights: @upload.CheckpointDeviceWeights::new(),
    queue: None,
    launch_count: 0,
    launches: [],
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
  self.prepare_layer(
    context,
    aot,
    Some(stream),
    checkpoint,
    root,
    maximum_device_bytes~,
    chunk_bytes~,
    limits~,
    counts~,
    reset_flag~,
    hidden~,
    output~,
  )
}

///|
fn GlmSparseLayer::prepare_layer(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  stream : @device.Stream?,
  checkpoint : @precision_materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming_safetensors.StreamingSafetensorsLimits,
  counts~ : @device.Allocation,
  reset_flag~ : @device.Allocation,
  hidden~ : @device.Allocation,
  output~ : @device.Allocation,
  input_positions? : @device.Allocation,
  page_ports? : @history_frame.SparseHistoryPagePorts,
  transient_scratch? : @device.Allocation,
  expert_scratch? : @device.Allocation,
) -> Unit raise {
  let frames = self.workspace_bytes() + self.state_bytes()
  guard stream is None || !self.ffn.is_tensor_parallel() else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let banks = self.ffn.bank_bytes()
  guard self.status is Fresh &&
    frames <= maximum_device_bytes &&
    banks <= maximum_device_bytes - frames else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.status = Failed
  let (bound, ffn_offset) = self.bound_weights(checkpoint)
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
    transient_scratch?=expert_scratch,
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
    input_positions?,
    page_ports?,
    transient_scratch?,
  )
  let launches = self.decoder.launches().to_owned()
  launches.append(self.ffn.launches)
  self.launches.append(launches)
  self.launch_count = launches.length()
  match stream {
    Some(stream) => {
      self.queue = Some(
        context.create_ordered_kernel_executor(stream, launches),
      )
      self.status = Ready
    }
    None => self.status = Prepared
  }
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
