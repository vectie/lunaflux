///|
/// Complete retained window attention through its pre-inverse-RoPE output.
/// Compression-zero layers only: compressed attention must not silently fall
/// back to a window-only result. Output projections remain downstream.
pub struct DeepSeekWindowAttention {
  priv spec : @spec.DeepSeekV4ModelSpec
  priv layer : Int
  priv inputs : DeepSeekAttentionInputs
  priv window : @window_frame.WindowSharedKvFrame
  priv sink : @upload.CheckpointDeviceWeights
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut state : State
}

///|
pub fn window_attention_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  maximum_rows~ : Int,
  maximum_state_bytes~ : Int64,
  attention_accumulation? : @precision.AttentionAccumulationLaw = OrderedFloat32,
) -> @precision.WindowSharedKvPrecision raise {
  guard layer >= 0 && layer < spec.layer_count() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  guard spec.compression_ratio(layer) == Some(0) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  @precision.WindowSharedKvPrecision::new(
    query_heads=spec.attention_head_count(),
    head_width=spec.head_dimension(),
    window_tokens=spec.sliding_window(),
    maximum_rows~,
    maximum_positions=spec.max_position_embeddings(),
    maximum_state_bytes~,
    // Original sparse-attention numerical boundary: F32 online state, BF16
    // unnormalized P per KV64 block, sink mass only in the final denominator.
    probability=OnlineBf16(block_tokens=64),
    accumulation=attention_accumulation,
  )
}

///|
pub fn bind_window_attention_sink(
  checkpoint : @materialize.PackedCheckpoint,
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
) -> @materialize.PackedCheckpointWeight raise {
  ignore(
    window_attention_plan(
      spec,
      layer~,
      maximum_rows=1,
      maximum_state_bytes=1048576L,
    ),
  )
  bind_attention_sink(checkpoint, spec, layer~)
}

///|
/// Sink parameters are shared by window-only and compressed attention.
pub fn bind_attention_sink(
  checkpoint : @materialize.PackedCheckpoint,
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
) -> @materialize.PackedCheckpointWeight raise {
  guard layer >= 0 && layer < spec.layer_count() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  checkpoint.bind_vector(
    representation=@format.Representation::new(
      format=F32,
      rows=1,
      columns=spec.attention_head_count(),
    ),
    payload_name="layers.\{layer}.attn.attn_sink",
  )
}

///|
pub fn DeepSeekWindowAttention::new(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  maximum_rows~ : Int,
  maximum_workspace_bytes~ : Int64,
  attention_accumulation? : @precision.AttentionAccumulationLaw = OrderedFloat32,
  rotary_product? : @precision.RotaryProductLaw = SeparateFloat32Products,
  projection_scaled_dot_law? : @precision.ScaledDotReductionLaw = DecodeOperandsFloat32,
  query_square_law? : @precision.SquareReductionLaw = VectorizedTreeFloat32(
    partitions=256,
    vector_width=1,
  ),
) -> Self raise {
  let plan = window_attention_plan(
    spec,
    layer~,
    maximum_rows~,
    maximum_state_bytes=maximum_workspace_bytes,
    attention_accumulation~,
  )
  {
    spec,
    layer,
    inputs: DeepSeekAttentionInputs::new(
      spec,
      layer~,
      maximum_rows~,
      maximum_workspace_bytes=maximum_workspace_bytes - plan.state_bytes(),
      rotary_product~,
      projection_scaled_dot_law~,
      query_square_law~,
    ),
    window: @window_frame.WindowSharedKvFrame::new(
      @window_source.window_shared_kv_source(
        plan,
        symbol_prefix="deepseek_l\{layer}_window",
      ),
    ),
    sink: @upload.CheckpointDeviceWeights::new(),
    launches: [],
    state: Fresh,
  }
}

///|
pub fn DeepSeekWindowAttention::source_bytes(self : Self) -> Bytes {
  self.inputs.source_bytes() + b"\n" + self.window.source_bytes()
}

///|
pub fn DeepSeekWindowAttention::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  self.inputs.required_device_bytes(checkpoint) +
  self.window.state_bytes() +
  bind_window_attention_sink(checkpoint, self.spec, layer=self.layer).buffer_bytes()
}

///|
pub fn DeepSeekWindowAttention::prepare(
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
  let sink = bind_window_attention_sink(checkpoint, self.spec, layer=self.layer)
  self.inputs.prepare(
    context,
    aot,
    checkpoint,
    root,
    counts~,
    positions~,
    hidden~,
    maximum_device_bytes=maximum_device_bytes -
      self.window.state_bytes() -
      sink.buffer_bytes(),
    chunk_bytes~,
    limits~,
  )
  self.sink.upload(
    context,
    checkpoint,
    root,
    [sink],
    maximum_device_bytes=sink.buffer_bytes(),
    chunk_bytes~,
    limits~,
  )
  let inputs = self.inputs.outputs()
  self.window.prepare(
    context,
    aot,
    counts~,
    reset_flag~,
    positions~,
    queries=inputs.queries,
    shared_kv=inputs.shared_kv,
    attention_sink=self.sink.allocation(0),
    maximum_state_bytes=self.window.state_bytes(),
  )
  self.launches.append(self.inputs.launches())
  self.launches.append(self.window.launches())
  self.state = Prepared
}

///|
pub fn DeepSeekWindowAttention::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
pub fn DeepSeekWindowAttention::output(self : Self) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.window.output()
}

///|
pub fn DeepSeekWindowAttention::append_descriptor(
  self : Self,
) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.window.append_descriptor()
}

///|
pub fn DeepSeekWindowAttention::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.launches.clear()
  self.window.close()
  self.sink.close()
  self.inputs.close()
  self.state = Closed
}
