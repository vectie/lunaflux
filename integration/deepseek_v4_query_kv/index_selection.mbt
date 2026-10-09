///|
pub fn index_selection_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  maximum_rows~ : Int,
  maximum_context~ : Int,
  maximum_workspace_bytes~ : Int64,
) -> @precision.CompressedIndexSelectionPrecision raise {
  guard layer >= 0 &&
    layer < spec.layer_count() &&
    maximum_context <= spec.max_position_embeddings() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  guard spec.compression_ratio(layer) == Some(4) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  @precision.CompressedIndexSelectionPrecision::new(
    head_count=spec.index_head_count(),
    head_width=spec.index_head_dimension(),
    compression_ratio=4,
    maximum_rows~,
    maximum_positions=maximum_context,
    top_k=spec.index_top_k(),
    maximum_workspace_bytes~,
  )
}

///|
/// Full-head single-rank learned indexer. Its 19 ordered effects share the
/// decoder's existing queue. Distributed score reduction is a separate plan.
pub struct DeepSeekLearnedIndexer {
  priv queries : DeepSeekIndexQueries
  priv cache : DeepSeekCompressedKvCache
  priv selection : @selection_frame.CompressedIndexSelectionFrame
  priv launches : Array[@device.OrderedKernelLaunch]
  priv descriptors : Array[@device.Allocation]
  priv mut state : State
}

///|
pub fn DeepSeekLearnedIndexer::new(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  maximum_rows~ : Int,
  maximum_context~ : Int,
  maximum_state_bytes~ : Int64,
  maximum_workspace_bytes~ : Int64,
) -> Self raise {
  let selection = index_selection_plan(
    spec,
    layer~,
    maximum_rows~,
    maximum_context~,
    maximum_workspace_bytes~,
  )
  let query = index_queries_plan(
    spec,
    layer~,
    maximum_rows~,
    maximum_workspace_bytes=maximum_workspace_bytes -
      selection.workspace_bytes(),
  )
  {
    queries: DeepSeekIndexQueries::new(
      spec,
      layer~,
      maximum_rows~,
      maximum_workspace_bytes=query.workspace_bytes(),
    ),
    cache: DeepSeekCompressedKvCache::new(
      spec,
      layer~,
      for_indexer=true,
      maximum_rows~,
      maximum_context~,
      maximum_state_bytes~,
      maximum_workspace_bytes=maximum_workspace_bytes -
        query.workspace_bytes() -
        selection.workspace_bytes(),
    ),
    selection: @selection_frame.CompressedIndexSelectionFrame::new(
      @selection_source.compressed_index_selection_source(
        selection,
        symbol_prefix="deepseek_l\{layer}_index_selection",
      ),
    ),
    launches: [],
    descriptors: [],
    state: Fresh,
  }
}

///|
pub fn DeepSeekLearnedIndexer::source_bytes(self : Self) -> Bytes {
  self.queries.source_bytes() +
  b"\n" +
  self.cache.source_bytes() +
  b"\n" +
  self.selection.source_bytes()
}

///|
pub fn DeepSeekLearnedIndexer::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  self.queries.required_device_bytes(checkpoint) +
  self.cache.required_device_bytes(checkpoint) +
  self.selection.workspace_bytes()
}

///|
pub fn DeepSeekLearnedIndexer::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  counts~ : @device.Allocation,
  positions~ : @device.Allocation,
  reset_flag~ : @device.Allocation,
  hidden~ : @device.Allocation,
  low_rank_queries~ : @device.Allocation,
  offsets~ : @device.Allocation,
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming.StreamingSafetensorsLimits,
) -> Unit raise {
  guard self.state is Fresh &&
    self.required_device_bytes(checkpoint) <= maximum_device_bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  self.state = Failed
  self.queries.prepare(
    context,
    aot,
    checkpoint,
    root,
    counts~,
    positions~,
    hidden~,
    low_rank_queries~,
    maximum_device_bytes=maximum_device_bytes -
      self.cache.required_device_bytes(checkpoint) -
      self.selection.workspace_bytes(),
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
  let queries = self.queries.outputs()
  let descriptors = self.cache.append_descriptors()
  self.selection.prepare(
    context,
    aot,
    counts~,
    positions~,
    cache_counts=self.cache.counts(),
    cache_append=descriptors[1],
    queries=queries.queries,
    head_weights=queries.head_weights,
    cache_values=self.cache.values(),
    offsets~,
    maximum_workspace_bytes=self.selection.workspace_bytes(),
  )
  self.launches.append(self.queries.launches())
  self.launches.append(self.cache.launches())
  self.launches.append(self.selection.launches())
  self.descriptors.append(descriptors)
  self.state = Prepared
}

///|
pub fn DeepSeekLearnedIndexer::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
pub fn DeepSeekLearnedIndexer::outputs(
  self : Self,
) -> @selection_frame.CompressedIndexSelectionOutputs raise {
  self.selection.outputs()
}

///|
pub fn DeepSeekLearnedIndexer::append_descriptors(
  self : Self,
) -> ArrayView[@device.Allocation] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.descriptors[:]
}

///|
pub fn DeepSeekLearnedIndexer::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.launches.clear()
  self.descriptors.clear()
  self.selection.close()
  self.cache.close()
  self.queries.close()
  self.state = Closed
}
