///|
/// Single-rank inverse rotary and learned output projections. Group ownership
/// remains explicit in the plan; a distributed decoder must add its collective.
pub struct DeepSeekAttentionOutput {
  priv spec : @spec.DeepSeekV4ModelSpec
  priv layer : Int
  priv plan : @precision.GroupedAttentionOutputPrecision
  priv frame : @output_frame.GroupedAttentionOutputFrame
  priv weights : @upload.CheckpointDeviceWeights
  priv mut state : State
}

///|
pub fn attention_output_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  maximum_rows~ : Int,
  maximum_workspace_bytes~ : Int64,
) -> @precision.GroupedAttentionOutputPrecision raise {
  let frequency = rotary_frequency_policy(spec, layer~)
  @precision.GroupedAttentionOutputPrecision::new(
    head_count=spec.attention_head_count(),
    head_width=spec.head_dimension(),
    rotary_width=spec.rope_head_dimension(),
    group_count=spec.output_group_count(),
    output_rank=spec.o_lora_rank(),
    hidden_width=spec.hidden_size(),
    maximum_rows~,
    maximum_positions=spec.max_position_embeddings(),
    frequency~,
    maximum_workspace_bytes~,
  )
}

///|
pub fn bind_attention_output_weights(
  checkpoint : @materialize.PackedCheckpoint,
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
) -> Array[@materialize.PackedCheckpointWeight] raise {
  let plan = attention_output_plan(
    spec,
    layer~,
    maximum_rows=1,
    maximum_workspace_bytes=1048576L,
  )
  let representations = plan.weight_representations()
  [
    checkpoint.bind_weight(
      representation=representations[0],
      payload_name="layers.\{layer}.attn.wo_a.weight",
      scales_name="layers.\{layer}.attn.wo_a.scale",
    ),
    checkpoint.bind_weight(
      representation=representations[1],
      payload_name="layers.\{layer}.attn.wo_b.weight",
      scales_name="layers.\{layer}.attn.wo_b.scale",
    ),
  ]
}

///|
pub fn DeepSeekAttentionOutput::new(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  maximum_rows~ : Int,
  maximum_workspace_bytes~ : Int64,
) -> Self raise {
  let plan = attention_output_plan(
    spec,
    layer~,
    maximum_rows~,
    maximum_workspace_bytes~,
  )
  {
    spec,
    layer,
    plan,
    frame: @output_frame.GroupedAttentionOutputFrame::new(
      @output_source.grouped_attention_output_source(
        plan,
        symbol_prefix="deepseek_l\{layer}_output",
      ),
    ),
    weights: @upload.CheckpointDeviceWeights::new(),
    state: Fresh,
  }
}

///|
pub fn DeepSeekAttentionOutput::source_bytes(self : Self) -> Bytes {
  self.frame.source_bytes()
}

///|
pub fn DeepSeekAttentionOutput::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  bind_attention_output_weights(checkpoint, self.spec, layer=self.layer).fold(
    init=self.plan.workspace_bytes(),
    (total, weight) => total + weight.buffer_bytes(),
  )
}

///|
pub fn DeepSeekAttentionOutput::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  counts~ : @device.Allocation,
  positions~ : @device.Allocation,
  attention~ : @device.Allocation,
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
    bind_attention_output_weights(checkpoint, self.spec, layer=self.layer),
    maximum_device_bytes=maximum_device_bytes - self.plan.workspace_bytes(),
    chunk_bytes~,
    limits~,
  )
  self.frame.prepare(
    context,
    aot,
    counts~,
    positions~,
    attention~,
    weights=[self.weights.allocation(0), self.weights.allocation(1)],
    maximum_workspace_bytes=self.plan.workspace_bytes(),
  )
  self.state = Prepared
}

///|
pub fn DeepSeekAttentionOutput::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.frame.launches()
}

///|
pub fn DeepSeekAttentionOutput::output(self : Self) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.frame.output()
}

///|
pub fn DeepSeekAttentionOutput::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.frame.close()
  self.weights.close()
  self.state = Closed
}
