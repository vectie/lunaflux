///|
/// Explicit semantic cut and effect ownership for attention ingress. This
/// describes what is materialized, not which device artifact is fastest.
pub struct AttentionIngressCutPlan {
  priv schedule : @luna_fusion_strategy.FusionSchedule
  priv qkv_operation : Int
}

///|
pub fn AttentionIngressCutPlan::new(
  schedule : @luna_fusion_strategy.FusionSchedule,
  qkv_operation~ : Int,
) -> AttentionIngressCutPlan raise FusionPlanError {
  if qkv_operation < 0 || qkv_operation > 2147483644 {
    raise InvalidFusionPlan(OperationOrder)
  }
  { schedule, qkv_operation, }
}

///|
/// An ingress module includes the KV-write part of the attention operation,
/// not its read/softmax part. With no ingress module the original attention
/// must still write KV; it cannot be replaced by a read-only attention kernel.
pub fn AttentionIngressCutPlan::module_span(
  self : AttentionIngressCutPlan,
) -> (Int, Int)? {
  match self.schedule {
    FullyFused => Some((self.qkv_operation, 4))
    ProducerSeparated => Some((self.qkv_operation + 1, 3))
    Unfused => None
  }
}

///|
pub fn AttentionIngressCutPlan::permits_readonly_attention(
  self : AttentionIngressCutPlan,
) -> Bool {
  self.schedule != Unfused
}

///|
/// (projected QKV, normalized QKV, positioned QKV) materialization boundaries.
pub fn AttentionIngressCutPlan::materialized_values(
  self : AttentionIngressCutPlan,
) -> (Bool, Bool, Bool) {
  match self.schedule {
    FullyFused => (false, false, true)
    ProducerSeparated => (true, false, true)
    Unfused => (true, true, true)
  }
}

///|
pub fn QkvQkNormPositionedRopePagedKvWritePlan::cut(
  self : QkvQkNormPositionedRopePagedKvWritePlan,
  schedule : @luna_fusion_strategy.FusionSchedule,
) -> AttentionIngressCutPlan raise FusionPlanError {
  if (schedule == FullyFused && !self.projected_exclusive) ||
    (schedule != Unfused && !self.normalized_exclusive) {
    raise InvalidFusionPlan(Dataflow)
  }
  { schedule, qkv_operation: self.qkv_operation.as_int(), }
}
