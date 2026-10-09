///|
/// The model builder selects recurrent semantics; device scheduling is absent.
pub fn recurrent_compute_plan(
  spec : @glm53_spec.Glm53FlashSpec,
  layer_index~ : Int,
  maximum_rows~ : Int,
  maximum_sequences~ : Int,
  cache_slots~ : Int,
  maximum_state_bytes~ : Int64,
) -> @precision_ir.RecurrentDeltaPrecision raise {
  guard spec.attention_kind(layer_index) is KimiDeltaAttention else {
    raise @precision_ir.PrecisionPlanError::IllegalPlacement
  }
  @precision_ir.RecurrentDeltaPrecision::new(
    head_count=spec.linear_head_count(),
    key_dimension=spec.linear_head_dimension(),
    value_dimension=spec.linear_head_dimension(),
    maximum_rows~,
    maximum_sequences~,
    cache_slots~,
    normalization_epsilon=0.000001,
    log_decay_lower_bound=spec.linear_gate_lower_bound(),
    maximum_state_bytes~,
  )
}
