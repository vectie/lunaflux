///|
/// The installed Flash router uses BF16 weights, F32 score arithmetic and a
/// true F32 rank-one correction vector. Selection policy stays in numeric IR.
pub fn routing_compute_plan(
  maximum_rows~ : Int,
  maximum_workspace_bytes~ : Int64,
  accuracy~ : @precision_ir.Accuracy,
) -> @precision_ir.MoeRoutingPrecision raise {
  let profile = @glm53_spec.Glm53FlashSpec::official()
  let representation = @precision_format.Representation::new(
    format=Bf16,
    rows=profile.routed_expert_count(),
    columns=profile.hidden_size(),
  )
  @precision_ir.MoeRoutingPrecision::new(
    weight=@precision_ir.plan_operand(
      role=Weight,
      representation~,
      placement=NativeOperand,
      supported_placements=[NativeOperand],
      accuracy~,
      memory_ceiling=representation.payload_bytes(),
    ),
    maximum_rows~,
    group_count=profile.expert_group_count(),
    group_score_topk=profile.expert_group_score_topk(),
    selected_group_count=profile.selected_expert_group_count(),
    selected_experts=profile.experts_per_token(),
    score=Sigmoid,
    normalize=profile.normalizes_selected_expert_probabilities(),
    scaling=profile.routed_scaling_factor(),
    maximum_workspace_bytes~,
  )
}

///|
pub fn bind_router(
  checkpoint : @precision_materialize.PackedCheckpoint,
  layer_index~ : Int,
) -> Array[@precision_materialize.PackedCheckpointWeight] raise {
  guard layer_index >= 3 && layer_index <= 45 else {
    raise @glm53_weights.Glm53NvFp4ExpertError::InvalidMoeLayer(layer_index)
  }
  let profile = @glm53_spec.Glm53FlashSpec::official()
  let prefix = "model.language_model.layers.\{layer_index}.mlp.gate."
  [
    checkpoint.bind_weight(
      representation=@precision_format.Representation::new(
        format=Bf16,
        rows=profile.routed_expert_count(),
        columns=profile.hidden_size(),
      ),
      payload_name=prefix + "weight",
    ),
    checkpoint.bind_vector(
      representation=@precision_format.Representation::new(
        format=F32,
        rows=1,
        columns=profile.routed_expert_count(),
      ),
      payload_name=prefix + "e_score_correction_bias",
    ),
  ]
}
