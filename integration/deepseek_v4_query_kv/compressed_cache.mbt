///|
pub fn compressed_cache_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  maximum_rows~ : Int,
  maximum_context~ : Int,
  maximum_state_bytes~ : Int64,
  maximum_workspace_bytes~ : Int64,
) -> @precision.RetainedRowsPrecision raise {
  guard layer >= 0 &&
    layer < spec.layer_count() &&
    maximum_rows > 0 &&
    maximum_context >= maximum_rows &&
    maximum_context <= spec.max_position_embeddings() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  guard spec.compression_ratio(layer) is Some(ratio) && ratio > 0 else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  @precision.RetainedRowsPrecision::new(
    word_width=spec.head_dimension(),
    position_stride=ratio,
    maximum_input_rows=(maximum_rows + ratio - 1) / ratio,
    maximum_stored_rows=maximum_context / ratio,
    maximum_state_bytes~,
    maximum_workspace_bytes~,
  )
}

///|
/// Checkpoint-connected attention pooling/precision plus a retained compressed
/// cache. Ten effects share the containing queue. Sparse indexing and the
/// ordinary sliding window remain separate consumers/owners.
pub struct DeepSeekCompressedKvCache {
  priv compressor : DeepSeekAttentionCompressor
  priv cache : @retained_frame.RetainedRowsFrame
  priv launches : Array[@device.OrderedKernelLaunch]
  priv descriptors : Array[@device.Allocation]
  priv mut state : State
}

///|
pub fn DeepSeekCompressedKvCache::new(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  maximum_rows~ : Int,
  maximum_context~ : Int,
  maximum_state_bytes~ : Int64,
  maximum_workspace_bytes~ : Int64,
) -> Self raise {
  let plan = compressed_cache_plan(
    spec,
    layer~,
    maximum_rows~,
    maximum_context~,
    maximum_state_bytes~,
    maximum_workspace_bytes~,
  )
  let state_bytes = plan
    .state_bytes()
    .fold(init=0L, (total, bytes) => total + bytes)
  {
    compressor: DeepSeekAttentionCompressor::new(
      spec,
      layer~,
      maximum_rows~,
      maximum_context~,
      maximum_state_bytes=maximum_state_bytes - state_bytes,
      maximum_workspace_bytes=maximum_workspace_bytes - plan.workspace_bytes(),
    ),
    cache: @retained_frame.RetainedRowsFrame::new(
      @retained_source.retained_rows_source(
        plan,
        symbol_prefix="deepseek_l\{layer}_compressed_history",
      ),
    ),
    launches: [],
    descriptors: [],
    state: Fresh,
  }
}

///|
pub fn DeepSeekCompressedKvCache::source_bytes(self : Self) -> Bytes {
  self.compressor.source_bytes() + b"\n" + self.cache.source_bytes()
}

///|
pub fn DeepSeekCompressedKvCache::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  self.compressor.required_device_bytes(checkpoint) +
  self.cache.required_device_bytes()
}

///|
pub fn DeepSeekCompressedKvCache::prepare(
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
  self.compressor.prepare(
    context,
    aot,
    checkpoint,
    root,
    counts~,
    positions~,
    reset_flag~,
    hidden~,
    maximum_device_bytes=maximum_device_bytes -
      self.cache.required_device_bytes(),
    chunk_bytes~,
    limits~,
  )
  self.cache.prepare(
    context,
    aot,
    reset_flag~,
    counts=self.compressor.output_counts(),
    positions=self.compressor.output_positions(),
    upstream_append=self.compressor.append_descriptor(),
    input=self.compressor.output(),
    maximum_device_bytes=self.cache.required_device_bytes(),
  )
  self.launches.append(self.compressor.launches())
  self.launches.append(self.cache.launches())
  self.descriptors.push(self.compressor.append_descriptor())
  self.descriptors.push(self.cache.append_descriptor())
  self.state = Prepared
}

///|
pub fn DeepSeekCompressedKvCache::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
pub fn DeepSeekCompressedKvCache::values(
  self : Self,
) -> @device.Allocation raise {
  self.cache.values()
}

///|
pub fn DeepSeekCompressedKvCache::counts(
  self : Self,
) -> @device.Allocation raise {
  self.cache.counts()
}

///|
pub fn DeepSeekCompressedKvCache::append_descriptors(
  self : Self,
) -> ArrayView[@device.Allocation] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.descriptors[:]
}

///|
pub fn DeepSeekCompressedKvCache::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.launches.clear()
  self.descriptors.clear()
  self.cache.close()
  self.compressor.close()
  self.state = Closed
}
