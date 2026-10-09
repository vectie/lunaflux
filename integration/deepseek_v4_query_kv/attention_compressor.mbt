///|
pub fn compressor_cache_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  maximum_input_rows~ : Int,
  maximum_workspace_bytes~ : Int64,
) -> @precision.RotaryCachePrecision raise {
  guard layer >= 0 &&
    layer < spec.layer_count() &&
    maximum_input_rows > 0 &&
    maximum_input_rows <= 65536 else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  guard spec.compression_ratio(layer) is Some(ratio) && ratio > 0 else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  @precision.RotaryCachePrecision::new(
    head_count=1,
    head_width=spec.head_dimension(),
    rotary_width=spec.rope_head_dimension(),
    maximum_rows=(maximum_input_rows + ratio - 1) / ratio,
    maximum_positions=spec.max_position_embeddings(),
    frequency=rotary_frequency_policy(spec, layer~),
    simulation_block_width=64,
    simulation_amax_floor=0.0001,
    simulation_scale=PowerOfTwo,
    maximum_workspace_bytes~,
  )
}

///|
/// Checkpoint-backed attention compressor through normalized pooling, group-
/// start rotary and prefix-only simulated E4M3. Indexer Hadamard/FP4 is a
/// different numerical operation and is never substituted by this owner.
pub struct DeepSeekAttentionCompressor {
  priv pool : DeepSeekCompressorPool
  priv transform : @rotary_frame.RotaryCacheFrame
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut state : State
}

///|
pub fn DeepSeekAttentionCompressor::new(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  maximum_rows~ : Int,
  maximum_state_bytes~ : Int64,
  maximum_workspace_bytes~ : Int64,
) -> Self raise {
  let cache = compressor_cache_plan(
    spec,
    layer~,
    maximum_input_rows=maximum_rows,
    maximum_workspace_bytes~,
  )
  {
    pool: DeepSeekCompressorPool::new(
      spec,
      layer~,
      maximum_rows~,
      maximum_state_bytes~,
      maximum_workspace_bytes=maximum_workspace_bytes - cache.workspace_bytes(),
    ),
    transform: @rotary_frame.RotaryCacheFrame::new(
      @rotary_source.rotary_cache_source(
        cache,
        symbol_prefix="deepseek_l\{layer}_compressed_kv",
      ),
    ),
    launches: [],
    state: Fresh,
  }
}

///|
pub fn DeepSeekAttentionCompressor::source_bytes(self : Self) -> Bytes {
  self.pool.source_bytes() + b"\n" + self.transform.source_bytes()
}

///|
pub fn DeepSeekAttentionCompressor::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  self.pool.required_device_bytes(checkpoint) + self.transform.workspace_bytes()
}

///|
pub fn DeepSeekAttentionCompressor::prepare(
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
    maximum_device_bytes=maximum_device_bytes - self.transform.workspace_bytes(),
    chunk_bytes~,
    limits~,
  )
  self.transform.prepare(
    context,
    aot,
    counts=self.pool.output_counts(),
    positions=self.pool.output_positions(),
    input=self.pool.output(),
    maximum_workspace_bytes=self.transform.workspace_bytes(),
  )
  self.launches.append(self.pool.launches())
  self.launches.append(self.transform.launches())
  self.state = Prepared
}

///|
pub fn DeepSeekAttentionCompressor::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
pub fn DeepSeekAttentionCompressor::output(
  self : Self,
) -> @device.Allocation raise {
  self.transform.output()
}

///|
pub fn DeepSeekAttentionCompressor::output_counts(
  self : Self,
) -> @device.Allocation raise {
  self.pool.output_counts()
}

///|
pub fn DeepSeekAttentionCompressor::output_positions(
  self : Self,
) -> @device.Allocation raise {
  self.pool.output_positions()
}

///|
pub fn DeepSeekAttentionCompressor::append_descriptor(
  self : Self,
) -> @device.Allocation raise {
  self.pool.append_descriptor()
}

///|
pub fn DeepSeekAttentionCompressor::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.launches.clear()
  self.transform.close()
  self.pool.close()
  self.state = Closed
}
