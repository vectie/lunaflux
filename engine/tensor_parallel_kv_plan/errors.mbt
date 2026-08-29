///|
/// Exact startup or lookup rule rejected by rank-local KV admission.
pub(all) enum TensorParallelKvPlanRule {
  BindingCount
  Rank
  WorldSize
  ModelIdentity
  ModelGeneration
  DeviceTarget
  DeviceOrdinal
  LayoutVersion
  KvHeadDivisibility
  RankArenaCeiling
  SizeOverflow
  FullArenaReplication
  PageIndex
  LayerIndex
  TokenIndex
  LocalKvHeadIndex
  HeadDimensionIndex
} derive(Debug, Eq)

///|
/// Payload-safe failure from startup admission or scalar address projection.
pub(all) suberror TensorParallelKvPlanError {
  RejectedTensorParallelKvPlan(rule~ : TensorParallelKvPlanRule, rank~ : Int)
} derive(Debug, Eq)

///|
fn rejected(
  rule : TensorParallelKvPlanRule,
  rank : Int,
) -> TensorParallelKvPlanError {
  RejectedTensorParallelKvPlan(rule~, rank~)
}
