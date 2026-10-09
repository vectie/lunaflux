///|
/// GLM's family-specific shape/clamp is translated once into the generic
/// precision/expert plan. No model-name decisions reach execution or lowering.
pub fn expert_compute_plan(
  layer_index~ : Int,
  expert_indices~ : ArrayView[Int],
  maximum_rows~ : Int,
  memory_ceiling~ : Int64,
  accuracy~ : @precision_ir.Accuracy,
) -> @precision_ir.ExpertMlpPrecision raise {
  let family = @glm53_weights.nvfp4_expert_weights(
    layer_index~,
    expert_indices~,
  )
  let operands = family[:3].map(weight => {
    @precision_ir.plan_operand(
      role=Weight,
      representation=weight.representation(),
      placement=DecodeInOperandTile,
      supported_placements=[DecodeInOperandTile],
      accuracy~,
      memory_ceiling~,
      tile_elements=32,
    )
  })
  @precision_ir.ExpertMlpPrecision::new(
    gate=operands[0],
    up=operands[1],
    down=operands[2],
    expert_count=288,
    local_experts=expert_indices,
    selected_experts=8,
    maximum_rows~,
    clamp=10.0,
    memory_ceiling~,
  )
}

///|
/// Upload packed weights into the executable plan's single expert bank.
/// Scratch is bounded and reused; allocation ownership stays with the caller.
/// Binding follows the plan's expert order, not an independently ordered list.
pub fn upload_expert_bank(
  context : @device.Context,
  checkpoint : @precision_materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  plan : @precision_ir.ExpertMlpPrecision,
  bank : @device.Allocation,
  layer_index~ : Int,
  chunk_bytes~ : Int,
  limits~ : @streaming_safetensors.StreamingSafetensorsLimits,
) -> Unit raise {
  guard chunk_bytes > 0 else { raise InvalidChunkSize }
  let indices = Array::makei(plan.local_experts().length(), i => {
    plan.local_experts()[i]
  })
  let weights = bind_experts(checkpoint, layer_index~, expert_indices=indices)
  guard weights.length() == plan.local_experts().length() * 3 else {
    raise InvalidDestinationCount
  }
  @packed_expert_mlp.upload_bank(
    context,
    checkpoint,
    root,
    plan,
    bank,
    weights~,
    chunk_bytes~,
    limits~,
  )
}
