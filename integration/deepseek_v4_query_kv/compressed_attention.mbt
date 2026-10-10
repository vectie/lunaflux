///|
/// Full-head compressed attention, before inverse rotary/output projection.
/// All owners contribute to one executor. Distributed index-score reduction
/// is not implied by this single-rank composition.
pub struct DeepSeekCompressedAttention {
  priv spec : @spec.DeepSeekV4ModelSpec
  priv layer : Int
  priv inputs : DeepSeekAttentionInputs
  priv cache : DeepSeekCompressedKvCache
  priv indexer : DeepSeekLearnedIndexer?
  priv attention : @compressed_window_frame.WindowCompressedKvFrame
  priv sink : @upload.CheckpointDeviceWeights
  priv launches : Array[@device.OrderedKernelLaunch]
  priv descriptors : Array[@device.Allocation]
  priv mut state : State
}

///|
pub fn DeepSeekCompressedAttention::new(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  maximum_rows~ : Int,
  maximum_context~ : Int,
  maximum_state_bytes~ : Int64,
  maximum_workspace_bytes~ : Int64,
) -> Self raise {
  let plan = compressed_attention_plan(
    spec,
    layer~,
    maximum_rows~,
    maximum_context~,
    maximum_state_bytes~,
  )
  let prefix_bytes = query_kv_plan(
      spec,
      maximum_rows~,
      maximum_workspace_bytes~,
    ).workspace_bytes() +
    rotary_kv_plan(spec, layer~, maximum_rows~, maximum_workspace_bytes~).workspace_bytes()
  let (cache_state, cache_workspace) = compressed_cache_sizes(
    spec, layer, maximum_rows, maximum_context, false, maximum_state_bytes, maximum_workspace_bytes,
  )
  let (index_state, index_workspace) = if plan.read_set()
    is @precision.CompressedRowReadSet::Learned(top_k=_) {
    let (state, workspace) = compressed_cache_sizes(
      spec, layer, maximum_rows, maximum_context, true, maximum_state_bytes, maximum_workspace_bytes,
    )
    (
      state,
      workspace +
      index_queries_plan(spec, layer~, maximum_rows~, maximum_workspace_bytes~).workspace_bytes() +
      index_selection_plan(
        spec,
        layer~,
        maximum_rows~,
        maximum_context~,
        maximum_workspace_bytes~,
      ).workspace_bytes(),
    )
  } else {
    (0L, 0L)
  }
  guard plan.state_bytes() + cache_state + index_state <= maximum_state_bytes &&
    prefix_bytes + cache_workspace + index_workspace <= maximum_workspace_bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  {
    spec,
    layer,
    inputs: DeepSeekAttentionInputs::new(
      spec,
      layer~,
      maximum_rows~,
      maximum_workspace_bytes=prefix_bytes,
    ),
    cache: DeepSeekCompressedKvCache::new(
      spec,
      layer~,
      maximum_rows~,
      maximum_context~,
      maximum_state_bytes=cache_state,
      maximum_workspace_bytes=cache_workspace,
    ),
    indexer: if index_state > 0L {
      Some(
        DeepSeekLearnedIndexer::new(
          spec,
          layer~,
          maximum_rows~,
          maximum_context~,
          maximum_state_bytes=index_state,
          maximum_workspace_bytes=index_workspace,
        ),
      )
    } else {
      None
    },
    attention: @compressed_window_frame.WindowCompressedKvFrame::new(
      @compressed_window_source.window_compressed_kv_source(
        plan,
        symbol_prefix="deepseek_l\{layer}_joint_attention",
      ),
    ),
    sink: @upload.CheckpointDeviceWeights::new(),
    launches: [],
    descriptors: [],
    state: Fresh,
  }
}

///|
pub fn DeepSeekCompressedAttention::source_bytes(self : Self) -> Bytes {
  self.inputs.source_bytes() +
  b"\n" +
  self.cache.source_bytes() +
  b"\n" +
  (match self.indexer {
    Some(indexer) => indexer.source_bytes() + b"\n"
    None => b""
  }) +
  self.attention.source_bytes()
}

///|
pub fn DeepSeekCompressedAttention::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  self.inputs.required_device_bytes(checkpoint) +
  self.cache.required_device_bytes(checkpoint) +
  (match self.indexer {
    Some(indexer) => indexer.required_device_bytes(checkpoint)
    None => 0L
  }) +
  self.attention.state_bytes() +
  bind_attention_sink(checkpoint, self.spec, layer=self.layer).buffer_bytes()
}

///|
pub fn DeepSeekCompressedAttention::prepare(
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
  self.inputs.prepare(
    context,
    aot,
    checkpoint,
    root,
    counts~,
    positions~,
    hidden~,
    maximum_device_bytes=self.inputs.required_device_bytes(checkpoint),
    chunk_bytes~,
    limits~,
  )
  self.cache.prepare(
    context,
    aot,
    checkpoint,
    root,
    counts~,
    positions~,
    reset_flag~,
    hidden~,
    maximum_device_bytes=self.cache.required_device_bytes(checkpoint),
    chunk_bytes~,
    limits~,
  )
  let inputs = self.inputs.outputs()
  let read_ports = match self.indexer {
    Some(indexer) => {
      indexer.prepare(
        context,
        aot,
        checkpoint,
        root,
        counts~,
        positions~,
        reset_flag~,
        hidden~,
        low_rank_queries=inputs.low_rank_queries,
        maximum_device_bytes=indexer.required_device_bytes(checkpoint),
        chunk_bytes~,
        limits~,
      )
      let selection = indexer.outputs()
      @compressed_window_frame.CompressedReadPorts::CacheRelative(
        counts=selection.row_counts,
        rows=selection.row_ids,
      )
    }
    _ => @compressed_window_frame.CompressedReadPorts::AllCausal
  }
  let sink = bind_attention_sink(checkpoint, self.spec, layer=self.layer)
  self.sink.upload(
    context,
    checkpoint,
    root,
    [sink],
    maximum_device_bytes=sink.buffer_bytes(),
    chunk_bytes~,
    limits~,
  )
  let cache_descriptors = self.cache.append_descriptors()
  self.attention.prepare(
    context,
    aot,
    counts~,
    positions~,
    reset_flag~,
    queries=inputs.queries,
    shared_kv=inputs.shared_kv,
    attention_sink=self.sink.allocation(0),
    compressed_values=self.cache.values(),
    compressed_counts=self.cache.counts(),
    compressed_append=cache_descriptors[1],
    read_ports~,
    maximum_state_bytes=self.attention.state_bytes(),
  )
  self.launches.append(self.inputs.launches())
  self.launches.append(self.cache.launches())
  self.descriptors.append(cache_descriptors)
  if self.indexer is Some(indexer) {
    self.launches.append(indexer.launches())
    self.descriptors.append(indexer.append_descriptors())
  }
  self.launches.append(self.attention.launches())
  self.descriptors.push(self.attention.append_descriptor())
  self.state = Prepared
}

///|
pub fn DeepSeekCompressedAttention::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
pub fn DeepSeekCompressedAttention::output(
  self : Self,
) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.attention.output()
}

///|
pub fn DeepSeekCompressedAttention::append_descriptors(
  self : Self,
) -> ArrayView[@device.Allocation] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.descriptors[:]
}

///|
pub fn DeepSeekCompressedAttention::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.launches.clear()
  self.descriptors.clear()
  self.attention.close()
  self.sink.close()
  if self.indexer is Some(indexer) {
    indexer.close()
  }
  self.cache.close()
  self.inputs.close()
  self.state = Closed
}
