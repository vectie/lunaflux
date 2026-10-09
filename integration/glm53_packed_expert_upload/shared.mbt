///|
pub fn shared_expert_compute_plan(
  layer_index~ : Int,
  maximum_rows~ : Int,
  memory_ceiling~ : Int64,
  accuracy~ : @precision_ir.Accuracy,
) -> @precision_ir.ExpertMlpPrecision raise {
  let operands = @glm53_weights.shared_expert_weights(layer_index~).map(weight => {
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
    expert_count=1,
    local_experts=[0],
    selected_experts=1,
    maximum_rows~,
    clamp=10.0,
    memory_ceiling~,
  )
}

///|
pub fn bind_shared_expert(
  checkpoint : @precision_materialize.PackedCheckpoint,
  layer_index~ : Int,
) -> Array[@precision_materialize.PackedCheckpointWeight] raise {
  @glm53_weights.shared_expert_weights(layer_index~).map(weight => {
    checkpoint.bind_weight(
      representation=weight.representation(),
      payload_name=weight.payload_name(),
    )
  })
}

///|
pub fn upload_shared_expert_bank(
  context : @device.Context,
  checkpoint : @precision_materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  plan : @precision_ir.ExpertMlpPrecision,
  bank : @device.Allocation,
  layer_index~ : Int,
  chunk_bytes~ : Int,
  limits~ : @streaming_safetensors.StreamingSafetensorsLimits,
) -> Unit raise {
  guard plan.geometry().3 == 1 && plan.local_experts()[:] == [0] else {
    raise InvalidDestinationCount
  }
  @packed_expert_mlp.upload_bank(
    context,
    checkpoint,
    root,
    plan,
    bank,
    weights=bind_shared_expert(checkpoint, layer_index~),
    chunk_bytes~,
    limits~,
  )
}
