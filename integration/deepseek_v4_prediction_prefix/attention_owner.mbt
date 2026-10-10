///|
priv enum AttentionState {
  Fresh
  Failed
  Prepared
  Closed
}

///|
/// Complete checkpoint-backed prediction attention through hidden-width output.
/// Both phase tables share one committed ring and one compact weight owner.
/// mHC input/output mixing and prediction FFN remain enclosing operations.
pub struct DeepSeekPredictionAttention {
  priv spec : @spec.DeepSeekV4ModelSpec
  priv stage : Int
  priv main : @frame.RowProjectionNormFrame
  priv main_rotary : @rotary_frame.RotaryCacheFrame
  priv query : @query_frame.QuantizedQueryKvFrame
  priv draft_rotary : @rotary_frame.RotaryKvFrame
  priv ring : @draft_frame.CommittedDraftKvFrame
  priv output : @output_frame.GroupedAttentionOutputFrame
  priv output_bytes : Int64
  priv weights : @upload.CheckpointDeviceWeights
  priv workspace : Int64
  priv prime : Array[@device.OrderedKernelLaunch]
  priv predict : Array[@device.OrderedKernelLaunch]
  priv mut state : AttentionState
}

///|
pub fn DeepSeekPredictionAttention::new(
  spec : @spec.DeepSeekV4ModelSpec,
  stage~ : Int,
  maximum_main_rows~ : Int,
  maximum_positions~ : Int,
  maximum_workspace_bytes~ : Int64,
) -> Self raise {
  require_prediction_stage(spec, stage)
  let prefix = "deepseek_prediction_s\{stage}"
  let main = @frame.RowProjectionNormFrame::new(
    @source.row_projection_norm_source(
      @precision.RowProjectionNormPrecision::new(
        input_width=spec.hidden_size(),
        output_width=spec.head_dimension(),
        maximum_rows=maximum_main_rows,
        normalization_epsilon=spec.rms_norm_epsilon(),
        activation_amax_floor=0.0001,
        maximum_workspace_bytes~,
      ),
      symbol_prefix=prefix + "_main_kv",
    ),
  )
  let frequency = @precision.RotaryFrequency::Base(theta=spec.rope_theta())
  let main_rotary = @rotary_frame.RotaryCacheFrame::new(
    @rotary_source.rotary_cache_source(
      @precision.RotaryCachePrecision::new(
        head_count=1,
        head_width=spec.head_dimension(),
        rotary_width=spec.rope_head_dimension(),
        maximum_rows=maximum_main_rows,
        maximum_positions~,
        frequency~,
        simulation_block_width=64,
        simulation_amax_floor=0.0001,
        simulation_scale=PowerOfTwo,
        maximum_workspace_bytes~,
      ),
      symbol_prefix=prefix + "_main_rotary",
    ),
  )
  let query = @query_frame.QuantizedQueryKvFrame::new(
    @query_source.quantized_query_kv_source(
      @precision.QuantizedQueryKvPrecision::new(
        hidden_width=spec.hidden_size(),
        query_rank=spec.q_lora_rank(),
        head_count=spec.attention_head_count(),
        head_width=spec.head_dimension(),
        maximum_rows=spec.dspark_block_size(),
        normalization_epsilon=spec.rms_norm_epsilon(),
        activation_amax_floor=0.0001,
        maximum_workspace_bytes~,
      ),
      symbol_prefix=prefix + "_draft_query",
    ),
  )
  let draft_rotary = @rotary_frame.RotaryKvFrame::new(
    @rotary_source.rotary_kv_source(
      @precision.RotaryKvPrecision::new(
        query_heads=spec.attention_head_count(),
        key_value_heads=1,
        head_width=spec.head_dimension(),
        rotary_width=spec.rope_head_dimension(),
        maximum_rows=spec.dspark_block_size(),
        maximum_positions~,
        frequency~,
        simulation_block_width=64,
        simulation_amax_floor=0.0001,
        simulation_scale=PowerOfTwo,
        maximum_workspace_bytes~,
      ),
      symbol_prefix=prefix + "_draft_rotary",
    ),
  )
  let p = prediction_attention_plan(
    spec,
    maximum_main_rows~,
    maximum_positions~,
    maximum_state_bytes=maximum_workspace_bytes,
  )
  let output_plan = prediction_attention_output_plan(
    spec,
    maximum_positions~,
    maximum_workspace_bytes~,
  )
  let output = @output_frame.GroupedAttentionOutputFrame::new(
    @output_source.grouped_attention_output_source(
      output_plan,
      symbol_prefix=prefix + "_output",
    ),
  )
  let workspace = main.plan().workspace_bytes() +
    main_rotary.workspace_bytes() +
    query.workspace_bytes() +
    draft_rotary.workspace_bytes() +
    p.state_bytes() +
    output.workspace_bytes()
  guard workspace <= maximum_workspace_bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  {
    spec,
    stage,
    main,
    main_rotary,
    query,
    draft_rotary,
    output,
    output_bytes: output_plan.frame_bytes()[2],
    ring: prediction_attention_frame(
      spec,
      maximum_main_rows~,
      maximum_positions~,
      maximum_state_bytes=p.state_bytes(),
      symbol_prefix=prefix + "_attention",
    ),
    weights: @upload.CheckpointDeviceWeights::new(),
    workspace,
    prime: [],
    predict: [],
    state: Fresh,
  }
}

///|
pub fn DeepSeekPredictionAttention::source_bytes(self : Self) -> Bytes {
  self.main.source_bytes() +
  b"\n" +
  self.main_rotary.source_bytes() +
  b"\n" +
  self.query.source_bytes() +
  b"\n" +
  self.draft_rotary.source_bytes() +
  b"\n" +
  self.ring.source_bytes() +
  b"\n" +
  self.output.source_bytes()
}

///|
pub fn DeepSeekPredictionAttention::workspace_bytes(
  self : Self,
  output_borrowed? : Bool = false,
) -> Int64 {
  self.workspace - (if output_borrowed { self.output_bytes } else { 0L })
}

///|
pub fn DeepSeekPredictionAttention::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
  output_borrowed? : Bool = false,
) -> Int64 raise {
  bind_prediction_attention_weights(checkpoint, self.spec, stage=self.stage).fold(
    init=self.workspace_bytes(output_borrowed~),
    (total, w) => total + w.buffer_bytes(),
  )
}

///|
pub fn DeepSeekPredictionAttention::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  main_counts~ : @device.Allocation,
  draft_counts~ : @device.Allocation,
  main_positions~ : @device.Allocation,
  draft_positions~ : @device.Allocation,
  main_hidden~ : @device.Allocation,
  draft_hidden~ : @device.Allocation,
  reset_flag~ : @device.Allocation,
  start_position~ : @device.Allocation,
  output? : @device.Allocation,
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming.StreamingSafetensorsLimits,
) -> Unit raise {
  guard self.state is Fresh && !main_counts.same_storage(draft_counts) else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let workspace = self.workspace_bytes(output_borrowed=output is Some(_))
  guard self.required_device_bytes(
      checkpoint,
      output_borrowed=output is Some(_),
    ) <=
    maximum_device_bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  if output is Some(buffer) {
    context.validate_allocation_region(
      buffer,
      offset=0L,
      byte_count=self.output_bytes,
      alignment=2L,
    )
  }
  let (_, _, main_rows) = self.main.plan().geometry()
  let draft_rows = self.spec.dspark_block_size()
  for
    (b, bytes, alignment) in [
      (main_counts, 20L, 4L),
      (draft_counts, 20L, 4L),
      (main_positions, main_rows.to_int64() * 4L, 4L),
      (draft_positions, draft_rows.to_int64() * 4L, 4L),
      (main_hidden, self.main.plan().input_bytes(), 2L),
      (
        draft_hidden,
        draft_rows.to_int64() * self.spec.hidden_size().to_int64() * 2L,
        2L,
      ),
      (reset_flag, 4L, 4L),
      (start_position, 4L, 4L),
    ] {
    context.validate_allocation_region(
      b,
      offset=0L,
      byte_count=bytes,
      alignment~,
    )
  }
  self.state = Failed
  self.weights.upload(
    context,
    checkpoint,
    root,
    bind_prediction_attention_weights(checkpoint, self.spec, stage=self.stage),
    maximum_device_bytes=maximum_device_bytes - workspace,
    chunk_bytes~,
    limits~,
  )
  self.main.prepare(
    context,
    aot,
    counts=main_counts,
    input=main_hidden,
    matrix=self.weights.allocation(3),
    norm=self.weights.allocation(4),
    maximum_workspace_bytes=self.main.plan().workspace_bytes(),
  )
  self.main_rotary.prepare(
    context,
    aot,
    counts=main_counts,
    positions=main_positions,
    input=self.main.output(),
    maximum_workspace_bytes=self.main_rotary.workspace_bytes(),
  )
  self.query.prepare(
    context,
    aot,
    counts=draft_counts,
    hidden=draft_hidden,
    weights=[
      self.weights.allocation(0),
      self.weights.allocation(1),
      self.weights.allocation(2),
      self.weights.allocation(3),
      self.weights.allocation(4),
    ],
    maximum_workspace_bytes=self.query.workspace_bytes(),
  )
  let q = self.query.outputs()
  self.draft_rotary.prepare(
    context,
    aot,
    counts=draft_counts,
    positions=draft_positions,
    queries=q.queries,
    shared_kv=q.shared_kv,
    maximum_workspace_bytes=self.draft_rotary.workspace_bytes(),
  )
  let operands = self.draft_rotary.outputs()
  self.ring.prepare(
    context,
    aot,
    main_counts~,
    draft_counts~,
    reset_flag~,
    start_position~,
    main_kv=self.main_rotary.output(),
    queries=operands.queries,
    draft_kv=operands.shared_kv,
    attention_sink=self.weights.allocation(5),
    maximum_state_bytes=self.ring.state_bytes(),
  )
  self.output.prepare(
    context,
    aot,
    counts=draft_counts,
    positions=draft_positions,
    attention=self.ring.output(),
    weights=[self.weights.allocation(6), self.weights.allocation(7)],
    output?,
    maximum_workspace_bytes=self.output.workspace_bytes(
      output_borrowed=output is Some(_),
    ),
  )
  self.prime.append(self.main.launches())
  self.prime.append(self.main_rotary.launches())
  self.prime.append(self.ring.prime_launches())
  self.predict.append(self.main.launches())
  self.predict.append(self.main_rotary.launches())
  self.predict.append(self.query.launches())
  self.predict.append(self.draft_rotary.launches())
  self.predict.append(self.ring.prediction_launches())
  self.predict.append(self.output.launches())
  self.state = Prepared
}

///|
pub fn DeepSeekPredictionAttention::prime_launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.prime[:]
}

///|
pub fn DeepSeekPredictionAttention::prediction_launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.predict[:]
}

///|
pub fn DeepSeekPredictionAttention::output(
  self : Self,
) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.output.output()
}

///|
pub fn DeepSeekPredictionAttention::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.prime.clear()
  self.predict.clear()
  self.output.close()
  self.ring.close()
  self.draft_rotary.close()
  self.query.close()
  self.main_rotary.close()
  self.main.close()
  self.weights.close()
  self.state = Closed
}
