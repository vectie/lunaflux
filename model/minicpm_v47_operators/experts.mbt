///|
fn routed_plan(
  spec : @model.Spec,
  maximum_rows~ : Int32,
) -> @precision.ExpertMlpPrecision raise {
  let geometry = spec.decoder()
  let experts = geometry.experts()
  let hidden = geometry.hidden_width()
  let intermediate = experts.intermediate_width
  let rows = maximum_rows.to_int64()
  // Exact per-layer extent for this BF16 layout, not whole-model admission.
  let bank = experts.routed.to_int64() *
    hidden.to_int64() *
    intermediate.to_int64() *
    6L
  let jobs = rows * experts.selected.to_int64()
  let ceiling = bank +
    jobs * (hidden + intermediate).to_int64() * 2L +
    rows * hidden.to_int64() * 6L +
    jobs * 8L +
    experts.routed.to_int64() * 4L +
    20L
  let operand = fn(height, width) raise {
    @precision.plan_operand(
      role=Weight,
      representation=@format.Representation::new(
        format=Bf16,
        rows=height,
        columns=width,
      ),
      placement=DecodeInOperandTile,
      supported_placements=[DecodeInOperandTile],
      accuracy=@precision.Accuracy::new(absolute=0.0, relative=0.0),
      memory_ceiling=ceiling,
      tile_elements=32,
    )
  }
  @precision.ExpertMlpPrecision::new(
    gate=operand(intermediate, hidden),
    up=operand(intermediate, hidden),
    down=operand(hidden, intermediate),
    expert_count=experts.routed,
    local_experts=Array::makei(experts.routed, i => i)[:],
    selected_experts=experts.selected,
    maximum_rows~,
    combine_rounding=Bf16EachAdd,
    memory_ceiling=ceiling,
  )
}

///|
/// One startup byte-range copy into the generic expert bank. Source offsets
/// include the safetensors header. No transpose, dequantization or full copy.
pub(all) struct ExpertUpload {
  shard_index : Int32
  file_offset : Int64
  bank_offset : Int64
  byte_length : Int64
}

///|
/// Split packed [expert,gate-then-up,out,in] by byte ranges while loading into
/// the existing per-expert gate/up/down bank. Only the destination bank persists.
pub fn DecoderOperators::expert_uploads(
  self : Self,
  weights : @weights.BoundWeights,
  layer~ : Int32,
) -> Array[ExpertUpload] raise {
  let prefix = "model.language_model.layers.\{layer}.mlp.experts."
  guard weights.find_tensor(prefix + "gate_up_proj") is Some(gate_up) &&
    weights.find_tensor(prefix + "down_proj") is Some(down) else {
    fail("missing MiniCPM expert tensor")
  }
  let plan = self.routed_experts()
  let (_, hidden, intermediate, _, _) = plan.geometry()
  let matrix = hidden.to_int64() * intermediate.to_int64() * 2L
  let uploads = []
  for ordinal, expert in plan.local_experts() {
    let destination = ordinal.to_int64() * plan.expert_stride()
    let source = expert.to_int64() * matrix
    uploads.push({
      shard_index: gate_up.shard_index,
      file_offset: gate_up.file_offset + source * 2L,
      bank_offset: destination + plan.weight_offsets()[0],
      byte_length: matrix,
    })
    uploads.push({
      shard_index: gate_up.shard_index,
      file_offset: gate_up.file_offset + source * 2L + matrix,
      bank_offset: destination + plan.weight_offsets()[1],
      byte_length: matrix,
    })
    uploads.push({
      shard_index: down.shard_index,
      file_offset: down.file_offset + source,
      bank_offset: destination + plan.weight_offsets()[2],
      byte_length: matrix,
    })
  }
  uploads
}
