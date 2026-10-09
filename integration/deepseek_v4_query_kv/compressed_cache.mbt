///|
pub fn compressed_cache_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  for_indexer? : Bool = false,
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
  guard spec.compression_ratio(layer) is Some(ratio) &&
    ratio > 0 &&
    (!for_indexer || ratio == 4) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  @precision.RetainedRowsPrecision::new(
    word_width=if for_indexer {
      spec.index_head_dimension()
    } else {
      spec.head_dimension()
    },
    position_stride=ratio,
    maximum_input_rows=(maximum_rows + ratio - 1) / ratio,
    maximum_stored_rows=maximum_context / ratio,
    maximum_state_bytes~,
    maximum_workspace_bytes~,
  )
}

///|
/// Checkpoint-connected attention pooling/precision plus a retained compressed
/// cache. Attention uses ten effects, indexer operands eleven, sharing the
/// containing queue. Learned index queries/top-k and the
/// ordinary sliding window remain separate consumers/owners.
pub struct DeepSeekCompressedKvCache {
  priv pool : DeepSeekCompressorPool
  priv transform : CompressorTransform
  priv cache : @retained_frame.RetainedRowsFrame
  priv launches : Array[@device.OrderedKernelLaunch]
  priv descriptors : Array[@device.Allocation]
  priv mut state : State
}

///|
pub fn DeepSeekCompressedKvCache::new(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  for_indexer? : Bool = false,
  maximum_rows~ : Int,
  maximum_context~ : Int,
  maximum_state_bytes~ : Int64,
  maximum_workspace_bytes~ : Int64,
) -> Self raise {
  let plan = compressed_cache_plan(
    spec,
    layer~,
    for_indexer~,
    maximum_rows~,
    maximum_context~,
    maximum_state_bytes~,
    maximum_workspace_bytes~,
  )
  let state_bytes = plan
    .state_bytes()
    .fold(init=0L, (total, bytes) => total + bytes)
  let transform = if for_indexer {
    CompressorTransform::Indexer(
      @indexer_frame.RotaryHadamardFp4Frame::new(
        @indexer_source.rotary_hadamard_fp4_source(
          indexer_cache_transform_plan(
            spec,
            layer~,
            maximum_rows~,
            maximum_workspace_bytes=maximum_workspace_bytes -
              plan.workspace_bytes(),
          ),
          symbol_prefix="deepseek_l\{layer}_indexer_kv",
        ),
      ),
    )
  } else {
    CompressorTransform::Attention(
      @rotary_frame.RotaryCacheFrame::new(
        @rotary_source.rotary_cache_source(
          compressor_cache_plan(
            spec,
            layer~,
            maximum_input_rows=maximum_rows,
            maximum_workspace_bytes=maximum_workspace_bytes -
              plan.workspace_bytes(),
          ),
          symbol_prefix="deepseek_l\{layer}_compressed_kv",
        ),
      ),
    )
  }
  {
    pool: DeepSeekCompressorPool::new(
      spec,
      layer~,
      for_indexer~,
      maximum_rows~,
      maximum_context~,
      maximum_state_bytes=maximum_state_bytes - state_bytes,
      maximum_workspace_bytes=maximum_workspace_bytes -
        plan.workspace_bytes() -
        transform.workspace_bytes(),
    ),
    transform,
    cache: @retained_frame.RetainedRowsFrame::new(
      @retained_source.retained_rows_source(
        plan,
        symbol_prefix="deepseek_l\{layer}_\{if for_indexer { "indexer" } else { "attention" }}_compressed_history",
      ),
    ),
    launches: [],
    descriptors: [],
    state: Fresh,
  }
}

///|
pub fn DeepSeekCompressedKvCache::source_bytes(self : Self) -> Bytes {
  self.pool.source_bytes() +
  b"\n" +
  self.transform.source_bytes() +
  b"\n" +
  self.cache.source_bytes()
}

///|
pub fn DeepSeekCompressedKvCache::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  self.pool.required_device_bytes(checkpoint) +
  self.transform.workspace_bytes() +
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
  self.pool.prepare(
    context,
    aot,
    checkpoint,
    root,
    counts~,
    positions~,
    reset_flag~,
    hidden~,
    maximum_device_bytes=maximum_device_bytes -
      self.cache.required_device_bytes() -
      self.transform.workspace_bytes(),
    chunk_bytes~,
    limits~,
  )
  self.transform.prepare(context, aot, self.pool)
  self.cache.prepare(
    context,
    aot,
    reset_flag~,
    counts=self.pool.output_counts(),
    positions=self.pool.output_positions(),
    upstream_append=self.pool.append_descriptor(),
    input=self.transform.output(),
    maximum_device_bytes=self.cache.required_device_bytes(),
  )
  self.launches.append(self.pool.launches())
  self.launches.append(self.transform.launches())
  self.launches.append(self.cache.launches())
  self.descriptors.push(self.pool.append_descriptor())
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
  self.transform.close()
  self.pool.close()
  self.state = Closed
}
