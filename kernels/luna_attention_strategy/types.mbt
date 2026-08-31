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
  Workspace
  Partition
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

///|
/// One contiguous key/value interval assigned to a decode partial reduction.
/// Intervals are ordered, disjoint, page aligned except for the terminal end,
/// and cover the complete admitted context envelope.
pub struct AttentionKeyValuePartition {
  priv split_index : Int
  priv begin_token : Int
  priv end_token : Int
} derive(Debug, Eq)

///|
/// Backend-neutral F32 partial-online-softmax workspace geometry.
///
/// Storage is three contiguous regions: maximums, denominators, then numerator
/// vectors. Backends may lower those regions differently, but may not change
/// their deterministic row/head/split indexing or merge order.
pub struct AttentionDecodeWorkspacePlan {
  priv maximum_rows : Int
  priv maximum_context_tokens : Int
  priv split_count : Int
  priv partial_count : Int64
  priv maximum_offset_bytes : Int64
  priv denominator_offset_bytes : Int64
  priv numerator_offset_bytes : Int64
  priv total_bytes : Int64
  priv partitions : ReadOnlyArray[AttentionKeyValuePartition]
} derive(Debug, Eq)
