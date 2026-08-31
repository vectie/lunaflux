///|
pub fn AttentionShape::dimensions(
  self : AttentionShape,
) -> (Int, Int, Int, Int) {
  (self.query_heads, self.kv_heads, self.head_dimension, self.tokens_per_page)
}

///|
pub fn AttentionPhasePlan::phase(
  self : AttentionPhasePlan,
) -> AttentionExecutionPhase {
  self.phase
}

///|
pub fn AttentionPhasePlan::traversal(
  self : AttentionPhasePlan,
) -> AttentionTraversal {
  self.traversal
}

///|
pub fn AttentionPhasePlan::reduction(
  self : AttentionPhasePlan,
) -> AttentionReduction {
  self.reduction
}

///|
pub fn AttentionPhasePlan::query_tile(self : AttentionPhasePlan) -> Int {
  self.query_tile
}

///|
pub fn AttentionPhasePlan::key_value_splits(self : AttentionPhasePlan) -> Int {
  self.key_value_splits
}

///|
pub fn AttentionDispatchPlan::shape(
  self : AttentionDispatchPlan,
) -> AttentionShape {
  self.shape
}

///|
pub fn AttentionDispatchPlan::prefill(
  self : AttentionDispatchPlan,
) -> AttentionPhasePlan {
  self.prefill
}

///|
pub fn AttentionDispatchPlan::decode(
  self : AttentionDispatchPlan,
) -> AttentionPhasePlan {
  self.decode
}
