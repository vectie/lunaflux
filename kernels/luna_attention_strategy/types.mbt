///|
pub(all) enum AttentionExecutionPhase {
  MixedCompatibility
  Prefill
  Decode
} derive(Debug, Eq)

///|
pub(all) enum AttentionTraversal {
  QueryTile
  KeyValueSplit
} derive(Debug, Eq)

///|
pub(all) enum AttentionReduction {
  OnlineSoftmax
  SplitOnlineSoftmaxMerge
} derive(Debug, Eq)

///|
pub(all) enum AttentionStrategyRule {
  Shape
  Phase
  Tile
  Split
} derive(Debug, Eq)

///|
pub(all) suberror AttentionStrategyError {
  InvalidAttentionStrategy(AttentionStrategyRule)
} derive(Debug, Eq)

///|
pub struct AttentionShape {
  priv query_heads : Int
  priv kv_heads : Int
  priv head_dimension : Int
  priv tokens_per_page : Int
} derive(Debug, Eq)

///|
pub struct AttentionPhasePlan {
  priv phase : AttentionExecutionPhase
  priv traversal : AttentionTraversal
  priv reduction : AttentionReduction
  priv query_tile : Int
  priv key_value_splits : Int
} derive(Debug, Eq)

///|
pub struct AttentionDispatchPlan {
  priv shape : AttentionShape
  priv prefill : AttentionPhasePlan
  priv decode : AttentionPhasePlan
} derive(Debug, Eq)
