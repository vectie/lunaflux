///|
/// Model geometry only. The generic pool owns numerical/state semantics;
/// rotation, cache conversion and sparse indexing are separate downstream work.
pub fn compressor_pool_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  for_indexer? : Bool = false,
  maximum_rows~ : Int,
  maximum_context? : Int,
  maximum_state_bytes~ : Int64,
  maximum_workspace_bytes~ : Int64,
) -> @precision.LearnedGatedPoolPrecision raise {
  guard layer >= 0 && layer < spec.layer_count() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let positions = maximum_context.or(spec.max_position_embeddings())
  guard positions >= maximum_rows && positions <= spec.max_position_embeddings() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  guard spec.compression_ratio(layer) is Some(ratio) &&
    ratio > 0 &&
    (!for_indexer || ratio == 4) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  @precision.LearnedGatedPoolPrecision::new(
    input_width=spec.hidden_size(),
    feature_width=if for_indexer {
      spec.index_head_dimension()
    } else {
      spec.head_dimension()
    },
    ratio~,
    preceding_window_overlap=ratio == 4,
    maximum_rows~,
    maximum_positions=positions,
    norm_epsilon=spec.rms_norm_epsilon(),
    maximum_state_bytes~,
    maximum_workspace_bytes~,
  )
}

///|
pub fn bind_compressor_pool_weights(
  checkpoint : @materialize.PackedCheckpoint,
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  for_indexer? : Bool = false,
) -> Array[@materialize.PackedCheckpointWeight] raise {
  let plan = compressor_pool_plan(
    spec,
    layer~,
    for_indexer~,
    maximum_rows=1,
    maximum_state_bytes=16777216L,
    maximum_workspace_bytes=1048576L,
  )
  let (hidden, width, ratio, overlap, _, _) = plan.geometry()
  let channels = width * (if overlap { 2 } else { 1 })
  let prefix = "layers.\{layer}.attn." +
    (if for_indexer { "indexer.compressor." } else { "compressor." })
  let matrix = @format.Representation::new(
    format=Bf16,
    rows=channels,
    columns=hidden,
  )
  [
    checkpoint.bind_weight(
      representation=matrix,
      payload_name=prefix + "wkv.weight",
    ),
    checkpoint.bind_weight(
      representation=matrix,
      payload_name=prefix + "wgate.weight",
    ),
    checkpoint.bind_weight(
      representation=@format.Representation::new(
        format=F32,
        rows=ratio,
        columns=channels,
      ),
      payload_name=prefix + "ape",
    ),
    checkpoint.bind_vector(
      representation=@format.Representation::new(
        format=Bf16,
        rows=1,
        columns=width,
      ),
      payload_name=prefix + "norm.weight",
    ),
  ]
}

///|
/// Checkpoint-connected learned pool, before rotary and compressed-cache commit.
/// Its launches join the containing decoder's existing ordered queue.
pub struct DeepSeekCompressorPool {
  priv spec : @spec.DeepSeekV4ModelSpec
  priv layer : Int
  priv indexer : Bool
  priv source : @pool_source.LearnedGatedPoolCudaSource
  priv frame : @pool_frame.LearnedGatedPoolFrame
  priv weights : @upload.CheckpointDeviceWeights
  priv mut state : State
}

///|
pub fn DeepSeekCompressorPool::new(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  for_indexer? : Bool = false,
  maximum_rows~ : Int,
  maximum_context? : Int,
  maximum_state_bytes~ : Int64,
  maximum_workspace_bytes~ : Int64,
) -> Self raise {
  let plan = compressor_pool_plan(
    spec,
    layer~,
    for_indexer~,
    maximum_rows~,
    maximum_context?,
    maximum_state_bytes~,
    maximum_workspace_bytes~,
  )
  let source = @pool_source.learned_gated_pool_source(
    plan,
    symbol_prefix="deepseek_l\{layer}_" +
      (if for_indexer { "indexer_pool" } else { "attention_pool" }),
  )
  {
    spec,
    layer,
    indexer: for_indexer,
    source,
    frame: @pool_frame.LearnedGatedPoolFrame::new(source),
    weights: @upload.CheckpointDeviceWeights::new(),
    state: Fresh,
  }
}

///|
pub fn DeepSeekCompressorPool::source_bytes(self : Self) -> Bytes {
  self.source.source_bytes()
}

///|
pub fn DeepSeekCompressorPool::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  bind_compressor_pool_weights(
    checkpoint,
    self.spec,
    layer=self.layer,
    for_indexer=self.indexer,
  ).fold(init=self.frame.required_device_bytes(), (total, weight) => {
    total + weight.buffer_bytes()
  })
}

///|
pub fn DeepSeekCompressorPool::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  counts~ : @device.Allocation,
  positions~ : @device.Allocation,
  reset_flag~ : @device.Allocation,
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
    bind_compressor_pool_weights(
      checkpoint,
      self.spec,
      layer=self.layer,
      for_indexer=self.indexer,
    ),
    maximum_device_bytes=maximum_device_bytes -
      self.frame.required_device_bytes(),
    chunk_bytes~,
    limits~,
  )
  self.frame.prepare(
    context,
    aot,
    counts~,
    positions~,
    reset_flag~,
    input=hidden,
    weights=[
      self.weights.allocation(0),
      self.weights.allocation(1),
      self.weights.allocation(2),
      self.weights.allocation(3),
    ],
    maximum_device_bytes=self.frame.required_device_bytes(),
  )
  self.state = Prepared
}

///|
pub fn DeepSeekCompressorPool::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  self.frame.launches()
}

///|
pub fn DeepSeekCompressorPool::output(self : Self) -> @device.Allocation raise {
  self.frame.output()
}

///|
pub fn DeepSeekCompressorPool::output_counts(
  self : Self,
) -> @device.Allocation raise {
  self.frame.output_counts()
}

///|
pub fn DeepSeekCompressorPool::output_positions(
  self : Self,
) -> @device.Allocation raise {
  self.frame.output_positions()
}

///|
pub fn DeepSeekCompressorPool::append_descriptor(
  self : Self,
) -> @device.Allocation raise {
  self.frame.append_descriptor()
}

///|
/// Close the containing queue before this owner, then caller-owned input ports.
pub fn DeepSeekCompressorPool::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.frame.close()
  self.weights.close()
  self.state = Closed
}
