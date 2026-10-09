///|
/// Translate hash-then-noaux selection into the common routing precision IR.
/// Both modes project the installed BF16 gate and compute F32 sqrt-softplus.
pub fn routing_compute_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  layer_index~ : Int,
  maximum_rows~ : Int,
  maximum_workspace_bytes~ : Int64,
  accuracy~ : @precision.Accuracy,
) -> @precision.MoeRoutingPrecision raise {
  guard layer_index >= 0 && layer_index < spec.layer_count() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let representation = @format.Representation::new(
    format=Bf16,
    rows=spec.routed_expert_count(),
    columns=spec.hidden_size(),
  )
  @precision.MoeRoutingPrecision::new(
    weight=@precision.plan_operand(
      role=Weight,
      representation~,
      placement=NativeOperand,
      supported_placements=[NativeOperand],
      accuracy~,
      memory_ceiling=representation.payload_bytes(),
    ),
    maximum_rows~,
    group_count=1,
    group_score_topk=1,
    selected_group_count=1,
    selected_experts=spec.experts_per_token(),
    score=SqrtSoftplus,
    selection=if layer_index < spec.hash_layer_count() {
      TokenHash(vocabulary_size=spec.vocabulary_size())
    } else {
      GroupedTopK
    },
    normalize=spec.normalizes_selected_expert_probabilities(),
    scaling=spec.routed_scaling_factor(),
    maximum_workspace_bytes~,
  )
}

///|
/// Hash layers have only the router matrix; later layers also bind bias.
/// I64 token tables use the existing checked startup narrowing/upload path.
pub fn bind_router(
  checkpoint : @materialize.PackedCheckpoint,
  spec : @spec.DeepSeekV4ModelSpec,
  layer_index~ : Int,
) -> Array[@materialize.PackedCheckpointWeight] raise {
  guard layer_index >= 0 && layer_index < spec.layer_count() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let prefix = "layers.\{layer_index}.ffn.gate."
  let result = [
    checkpoint.bind_weight(
      representation=@format.Representation::new(
        format=Bf16,
        rows=spec.routed_expert_count(),
        columns=spec.hidden_size(),
      ),
      payload_name=prefix + "weight",
    ),
  ]
  if layer_index >= spec.hash_layer_count() {
    result.push(
      checkpoint.bind_vector(
        representation=@format.Representation::new(
          format=F32,
          rows=1,
          columns=spec.routed_expert_count(),
        ),
        payload_name=prefix + "bias",
      ),
    )
  }
  result
}
