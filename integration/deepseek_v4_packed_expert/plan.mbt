///|
fn activation_plan(
  rows : Int,
  columns : Int,
  accuracy : @precision.Accuracy,
  memory_ceiling : Int64,
) -> @precision.QuantizationPlan raise {
  @precision.plan_operand(
    role=Activation,
    representation=@format.Representation::new(
      format=MxFp8E4M3,
      rows~,
      columns~,
      grouping=Block(rows=1, columns=128),
    ),
    placement=QuantizeBeforeConsumer,
    supported_placements=[QuantizeBeforeConsumer],
    accuracy~,
    memory_ceiling~,
  )
  .quantization()
  .with_amax_floor(0.0001)
}

///|
/// Translate the reference expert's numeric order into generic immutable IR.
/// Dynamic FP8 is applied before both projections; routing weights multiply
/// the F32 SwiGLU product before its single BF16 rounding and down quantization.
pub fn expert_compute_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  layer_index~ : Int,
  expert_indices~ : ArrayView[Int],
  maximum_rows~ : Int,
  memory_ceiling~ : Int64,
  accuracy~ : @precision.Accuracy,
) -> @precision.ExpertMlpPrecision raise {
  block_expert_compute_plan(
    spec,
    block=base_block(spec, layer_index),
    expert_indices~,
    maximum_rows~,
    memory_ceiling~,
    accuracy~,
  )
}

///|
pub fn block_expert_compute_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  block~ : @spec.DeepSeekV4BlockAddress,
  expert_indices~ : ArrayView[Int],
  maximum_rows~ : Int,
  memory_ceiling~ : Int64,
  accuracy~ : @precision.Accuracy,
) -> @precision.ExpertMlpPrecision raise {
  let family = @weights.packed_block_expert_weights(
    spec,
    block~,
    expert_indices~,
  )
  compute_plan(
    spec,
    family,
    expert_count=spec.routed_expert_count(),
    selected_experts=spec.experts_per_token(),
    expert_indices~,
    maximum_rows~,
    memory_ceiling~,
    accuracy~,
  )
}

///|
fn compute_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  family : ArrayView[@weights.DeepSeekV4PackedExpertWeight],
  expert_count~ : Int,
  selected_experts~ : Int,
  expert_indices~ : ArrayView[Int],
  maximum_rows~ : Int,
  memory_ceiling~ : Int64,
  accuracy~ : @precision.Accuracy,
) -> @precision.ExpertMlpPrecision raise {
  let operands = family[:3].map(weight => {
    @precision.plan_operand(
      role=Weight,
      representation=weight.representation(),
      placement=DecodeInOperandTile,
      supported_placements=[DecodeInOperandTile],
      accuracy~,
      memory_ceiling~,
      tile_elements=32,
    )
  })
  let jobs = maximum_rows.to_int64() * selected_experts.to_int64()
  guard maximum_rows > 0 && jobs <= 2147483647L else {
    raise @precision.PrecisionPlanError::ArithmeticOverflow
  }
  @precision.ExpertMlpPrecision::new(
    gate=operands[0],
    up=operands[1],
    down=operands[2],
    expert_count~,
    local_experts=expert_indices,
    selected_experts~,
    maximum_rows~,
    clamp=spec.swiglu_limit(),
    activation_rounding=F32Product,
    routing_placement=BeforeDownInputRound,
    activation_quantization=(
      activation_plan(
        maximum_rows,
        spec.hidden_size(),
        accuracy,
        memory_ceiling,
      ),
      activation_plan(
        jobs.to_int(),
        spec.moe_intermediate_size(),
        accuracy,
        memory_ceiling,
      ),
    ),
    memory_ceiling~,
  )
}

///|
/// Same executable packed MLP, with a single all-row expert and unit scores.
/// Routing buffers are initialized once; their constant values are reusable.
pub fn shared_expert_compute_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  layer_index~ : Int,
  maximum_rows~ : Int,
  memory_ceiling~ : Int64,
  accuracy~ : @precision.Accuracy,
) -> @precision.ExpertMlpPrecision raise {
  block_shared_expert_compute_plan(
    spec,
    block=base_block(spec, layer_index),
    maximum_rows~,
    memory_ceiling~,
    accuracy~,
  )
}

///|
pub fn block_shared_expert_compute_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  block~ : @spec.DeepSeekV4BlockAddress,
  maximum_rows~ : Int,
  memory_ceiling~ : Int64,
  accuracy~ : @precision.Accuracy,
) -> @precision.ExpertMlpPrecision raise {
  compute_plan(
    spec,
    @weights.packed_block_shared_expert_weights(spec, block~),
    expert_count=1,
    selected_experts=1,
    expert_indices=[0],
    maximum_rows~,
    memory_ceiling~,
    accuracy~,
  )
}

///|
pub fn bind_shared_expert(
  checkpoint : @materialize.PackedCheckpoint,
  spec : @spec.DeepSeekV4ModelSpec,
  layer_index~ : Int,
) -> Array[@materialize.PackedCheckpointWeight] raise {
  bind_block_shared_expert(
    checkpoint,
    spec,
    block=base_block(spec, layer_index),
  )
}

///|
pub fn bind_block_shared_expert(
  checkpoint : @materialize.PackedCheckpoint,
  spec : @spec.DeepSeekV4ModelSpec,
  block~ : @spec.DeepSeekV4BlockAddress,
) -> Array[@materialize.PackedCheckpointWeight] raise {
  @weights.packed_block_shared_expert_weights(spec, block~).map(weight => {
    checkpoint.bind_weight(
      representation=weight.representation(),
      payload_name=weight.payload_name(),
      scales_name=weight.scales_name(),
    )
  })
}

///|
pub fn upload_shared_expert_bank(
  context : @device.Context,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  spec : @spec.DeepSeekV4ModelSpec,
  plan : @precision.ExpertMlpPrecision,
  bank : @device.Allocation,
  layer_index~ : Int,
  chunk_bytes~ : Int,
  limits~ : @streaming.StreamingSafetensorsLimits,
) -> Unit raise {
  guard plan.geometry().3 == 1 &&
    plan.geometry().4 == 1 &&
    plan.local_experts().length() == 1 &&
    plan.local_experts()[0] == 0 else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  @expert.upload_bank(
    context,
    checkpoint,
    root,
    plan,
    bank,
    weights=bind_shared_expert(checkpoint, spec, layer_index~),
    chunk_bytes~,
    limits~,
  )
}

///|
pub fn bind_experts(
  checkpoint : @materialize.PackedCheckpoint,
  spec : @spec.DeepSeekV4ModelSpec,
  layer_index~ : Int,
  expert_indices~ : ArrayView[Int],
) -> Array[@materialize.PackedCheckpointWeight] raise {
  bind_block_experts(
    checkpoint,
    spec,
    block=base_block(spec, layer_index),
    expert_indices~,
  )
}

///|
pub fn bind_block_experts(
  checkpoint : @materialize.PackedCheckpoint,
  spec : @spec.DeepSeekV4ModelSpec,
  block~ : @spec.DeepSeekV4BlockAddress,
  expert_indices~ : ArrayView[Int],
) -> Array[@materialize.PackedCheckpointWeight] raise {
  @weights.packed_block_expert_weights(spec, block~, expert_indices~).map(weight => {
    checkpoint.bind_weight(
      representation=weight.representation(),
      payload_name=weight.payload_name(),
      scales_name=weight.scales_name(),
    )
  })
}

///|
pub fn upload_expert_bank(
  context : @device.Context,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  spec : @spec.DeepSeekV4ModelSpec,
  plan : @precision.ExpertMlpPrecision,
  bank : @device.Allocation,
  layer_index~ : Int,
  chunk_bytes~ : Int,
  limits~ : @streaming.StreamingSafetensorsLimits,
) -> Unit raise {
  let indices = Array::makei(plan.local_experts().length(), i => {
    plan.local_experts()[i]
  })
  @expert.upload_bank(
    context,
    checkpoint,
    root,
    plan,
    bank,
    weights=bind_experts(checkpoint, spec, layer_index~, expert_indices=indices),
    chunk_bytes~,
    limits~,
  )
}
