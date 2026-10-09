///|
pub fn index_queries_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  maximum_rows~ : Int,
  maximum_workspace_bytes~ : Int64,
) -> @precision.LearnedIndexQueriesPrecision raise {
  guard layer >= 0 &&
    layer < spec.layer_count() &&
    spec.compression_ratio(layer) == Some(4) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  @precision.LearnedIndexQueriesPrecision::new(
    hidden_width=spec.hidden_size(),
    query_rank=spec.q_lora_rank(),
    head_count=spec.index_head_count(),
    @precision.RotaryHadamardFp4Precision::new(
      head_count=spec.index_head_count(),
      head_width=spec.index_head_dimension(),
      rotary_width=spec.rope_head_dimension(),
      maximum_rows~,
      maximum_positions=spec.max_position_embeddings(),
      frequency=rotary_frequency_policy(spec, layer~),
      simulation_block_width=32,
      maximum_workspace_bytes~,
    ),
    maximum_workspace_bytes~,
  )
}

///|
/// Full single-rank head domain. Query-B stays packed, head-weight bank BF16.
pub fn bind_index_queries_weights(
  checkpoint : @materialize.PackedCheckpoint,
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
) -> Array[@materialize.PackedCheckpointWeight] raise {
  let plan = index_queries_plan(
    spec,
    layer~,
    maximum_rows=1,
    maximum_workspace_bytes=1048576L,
  )
  let prefix = "layers.\{layer}.attn.indexer."
  [
    checkpoint.bind_weight(
      representation=plan.query_weight(),
      payload_name=prefix + "wq_b.weight",
      scales_name=prefix + "wq_b.scale",
    ),
    checkpoint.bind_weight(
      representation=@format.Representation::new(
        format=Bf16,
        rows=spec.index_head_count(),
        columns=spec.hidden_size(),
      ),
      payload_name=prefix + "weights_proj.weight",
    ),
  ]
}

///|
pub struct DeepSeekIndexQueries {
  priv spec : @spec.DeepSeekV4ModelSpec
  priv layer : Int
  priv frame : @index_queries_frame.LearnedIndexQueriesFrame
  priv weights : @upload.CheckpointDeviceWeights
  priv mut state : State
}

///|
pub fn DeepSeekIndexQueries::new(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  maximum_rows~ : Int,
  maximum_workspace_bytes~ : Int64,
) -> Self raise {
  let plan = index_queries_plan(
    spec,
    layer~,
    maximum_rows~,
    maximum_workspace_bytes~,
  )
  {
    spec,
    layer,
    frame: @index_queries_frame.LearnedIndexQueriesFrame::new(
      @index_queries_source.learned_index_queries_source(
        plan,
        symbol_prefix="deepseek_l\{layer}_index_queries",
      ),
    ),
    weights: @upload.CheckpointDeviceWeights::new(),
    state: Fresh,
  }
}

///|
pub fn DeepSeekIndexQueries::source_bytes(self : Self) -> Bytes {
  self.frame.source_bytes()
}

///|
pub fn DeepSeekIndexQueries::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  bind_index_queries_weights(checkpoint, self.spec, layer=self.layer).fold(
    init=self.frame.workspace_bytes(),
    (total, weight) => total + weight.buffer_bytes(),
  )
}

///|
pub fn DeepSeekIndexQueries::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  counts~ : @device.Allocation,
  positions~ : @device.Allocation,
  hidden~ : @device.Allocation,
  low_rank_queries~ : @device.Allocation,
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
    bind_index_queries_weights(checkpoint, self.spec, layer=self.layer),
    maximum_device_bytes=maximum_device_bytes - self.frame.workspace_bytes(),
    chunk_bytes~,
    limits~,
  )
  self.frame.prepare(
    context,
    aot,
    counts~,
    positions~,
    hidden~,
    low_rank_queries~,
    weights=[self.weights.allocation(0), self.weights.allocation(1)],
    maximum_workspace_bytes=self.frame.workspace_bytes(),
  )
  self.state = Prepared
}

///|
pub fn DeepSeekIndexQueries::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  self.frame.launches()
}

///|
pub fn DeepSeekIndexQueries::outputs(
  self : Self,
) -> @index_queries_frame.LearnedIndexQueryOutputs raise {
  self.frame.outputs()
}

///|
pub fn DeepSeekIndexQueries::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.frame.close()
  self.weights.close()
  self.state = Closed
}
