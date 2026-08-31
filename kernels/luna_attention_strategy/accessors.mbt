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

///|
pub fn AttentionKeyValuePartition::split_index(
  self : AttentionKeyValuePartition,
) -> Int {
  self.split_index
}

///|
pub fn AttentionKeyValuePartition::token_range(
  self : AttentionKeyValuePartition,
) -> (Int, Int) {
  (self.begin_token, self.end_token)
}

///|
pub fn AttentionDecodeWorkspacePlan::maximum_rows(
  self : AttentionDecodeWorkspacePlan,
) -> Int {
  self.maximum_rows
}

///|
pub fn AttentionDecodeWorkspacePlan::maximum_context_tokens(
  self : AttentionDecodeWorkspacePlan,
) -> Int {
  self.maximum_context_tokens
}

///|
pub fn AttentionDecodeWorkspacePlan::split_count(
  self : AttentionDecodeWorkspacePlan,
) -> Int {
  self.split_count
}

///|
pub fn AttentionDecodeWorkspacePlan::partial_count(
  self : AttentionDecodeWorkspacePlan,
) -> Int64 {
  self.partial_count
}

///|
pub fn AttentionDecodeWorkspacePlan::region_offsets_bytes(
  self : AttentionDecodeWorkspacePlan,
) -> (Int64, Int64, Int64) {
  (
    self.maximum_offset_bytes,
    self.denominator_offset_bytes,
    self.numerator_offset_bytes,
  )
}

///|
pub fn AttentionDecodeWorkspacePlan::total_bytes(
  self : AttentionDecodeWorkspacePlan,
) -> Int64 {
  self.total_bytes
}

///|
pub fn AttentionDecodeWorkspacePlan::partitions(
  self : AttentionDecodeWorkspacePlan,
) -> ReadOnlyArray[AttentionKeyValuePartition] {
  self.partitions
}
