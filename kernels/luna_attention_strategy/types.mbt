///|
pub(all) enum AttentionExecutionPhase {
  MixedCompatibility
  Prefill
  Decode
} derive(Debug, Eq)

///|
pub extend AttentionExecutionPhase with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionExecutionPhase with Eq::{not_equal, equal}

///|
pub(all) enum AttentionTraversal {
  QueryTile
  KeyValueSplit
} derive(Debug, Eq)

///|
pub extend AttentionTraversal with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionTraversal with Eq::{not_equal, equal}

///|
pub(all) enum AttentionReduction {
  OnlineSoftmax
  SplitOnlineSoftmaxMerge
} derive(Debug, Eq)

///|
pub extend AttentionReduction with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionReduction with Eq::{not_equal, equal}

///|
pub(all) enum AttentionStrategyRule {
  Shape
  Phase
  Tile
  Split
  Workspace
  Partition
  Capability
  Problem
  Measurement
  Candidate
  Backend
  Target
  Duplicate
  Conflict
} derive(Debug, Eq)

///|
pub extend AttentionStrategyRule with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionStrategyRule with Eq::{not_equal, equal}

///|
pub(all) suberror AttentionStrategyError {
  InvalidAttentionStrategy(AttentionStrategyRule)
} derive(Debug, Eq)

///|
pub extend AttentionStrategyError with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionStrategyError with Eq::{not_equal, equal}

///|
pub struct AttentionShape {
  priv query_heads : Int32
  priv kv_heads : Int32
  priv head_dimension : Int32
  priv tokens_per_page : Int32
} derive(Debug, Eq)

///|
pub extend AttentionShape with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionShape with Eq::{not_equal, equal}

///|
pub struct AttentionPhasePlan {
  priv phase : AttentionExecutionPhase
  priv traversal : AttentionTraversal
  priv reduction : AttentionReduction
  priv query_tile : Int32
  priv key_value_splits : Int32
} derive(Debug, Eq)

///|
pub extend AttentionPhasePlan with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionPhasePlan with Eq::{not_equal, equal}

///|
pub struct AttentionDispatchPlan {
  priv shape : AttentionShape
  priv prefill : AttentionPhasePlan
  priv decode : AttentionPhasePlan
} derive(Debug, Eq)

///|
pub extend AttentionDispatchPlan with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionDispatchPlan with Eq::{not_equal, equal}

///|
/// One contiguous key/value interval assigned to a decode partial reduction.
/// Intervals are ordered, disjoint, page aligned except for the terminal end,
/// and cover the complete admitted context envelope.
pub struct AttentionKeyValuePartition {
  priv split_index : Int32
  priv begin_token : Int32
  priv end_token : Int32
} derive(Debug, Eq)

///|
pub extend AttentionKeyValuePartition with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend AttentionKeyValuePartition with Eq::{not_equal, equal}

///|
/// Backend-neutral F32 partial-online-softmax workspace geometry.
///
/// Storage is three contiguous regions: maximums, denominators, then numerator
/// vectors. Backends may lower those regions differently, but may not change
/// their deterministic row/head/split indexing or merge order.
pub struct AttentionDecodeWorkspacePlan {
  priv maximum_rows : Int32
  priv maximum_context_tokens : Int32
  priv split_count : Int32
  priv partial_count : Int64
  priv maximum_offset_bytes : Int64
  priv denominator_offset_bytes : Int64
  priv numerator_offset_bytes : Int64
  priv total_bytes : Int64
  priv partitions : ReadOnlyArray[AttentionKeyValuePartition]
} derive(Debug, Eq)

///|
pub extend AttentionDecodeWorkspacePlan with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend AttentionDecodeWorkspacePlan with Eq::{not_equal, equal}
