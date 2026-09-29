///|
/// Explicit semantic cut and effect ownership for attention ingress. This
/// describes what is materialized, not which device artifact is fastest.
pub struct AttentionIngressCutPlan {
  priv schedule : @luna_fusion_strategy.FusionSchedule
  priv qkv_operation : Int
  priv partition : @fusion_regions.Partition
}

///|
pub fn AttentionIngressCutPlan::new(
  schedule : @luna_fusion_strategy.FusionSchedule,
  qkv_operation~ : Int,
) -> AttentionIngressCutPlan raise FusionPlanError {
  if qkv_operation < 0 || qkv_operation > 2147483644 {
    raise InvalidFusionPlan(OperationOrder)
  }
  ingress_cut(
    schedule,
    qkv_operation~,
    projected_observable=false,
    normalized_observable=false,
  )
}

///|
/// An ingress module includes the KV-write part of the attention operation,
/// not its read/softmax part. With no ingress module the original attention
/// must still write KV; it cannot be replaced by a read-only attention kernel.
pub fn AttentionIngressCutPlan::module_span(
  self : AttentionIngressCutPlan,
) -> (Int, Int)? {
  for region in self.partition.regions() {
    let (start, count) = region.span()
    if count > 1 && start + count == 4 {
      return Some((self.qkv_operation + start, count))
    }
  }
  None
}

///|
pub fn AttentionIngressCutPlan::permits_readonly_attention(
  self : AttentionIngressCutPlan,
) -> Bool {
  self.module_span() is Some(_)
}

///|
/// (projected QKV, normalized QKV, positioned QKV) materialization boundaries.
pub fn AttentionIngressCutPlan::materialized_values(
  self : AttentionIngressCutPlan,
) -> (Bool, Bool, Bool) {
  let regions = self.partition.regions()
  (
    regions.any(region => region.exports().contains(0)),
    regions.any(region => region.exports().contains(1)),
    regions.any(region => region.exports().contains(2)),
  )
}

///|
pub fn QkvQkNormPositionedRopePagedKvWritePlan::cut(
  self : QkvQkNormPositionedRopePagedKvWritePlan,
  schedule : @luna_fusion_strategy.FusionSchedule,
) -> AttentionIngressCutPlan raise FusionPlanError {
  ingress_cut(
    schedule,
    qkv_operation=self.qkv_operation.as_int(),
    projected_observable=!self.projected_exclusive,
    normalized_observable=!self.normalized_exclusive,
  )
}
