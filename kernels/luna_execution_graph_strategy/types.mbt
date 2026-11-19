///|
pub(all) enum ExecutionGraphPhase {
  PrefillGraph
  DecodeGraph
  MixedGraph
} derive(Debug, Eq)

///|
pub extend ExecutionGraphPhase with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ExecutionGraphPhase with Eq::{not_equal, equal}

///|
pub(all) enum ExecutionGraphBucketRule {
  Shape
  Limit
  Capacity
} derive(Debug, Eq)

///|
pub extend ExecutionGraphBucketRule with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend ExecutionGraphBucketRule with Eq::{not_equal, equal}

///|
pub(all) suberror ExecutionGraphBucketError {
  InvalidExecutionGraphBucket(ExecutionGraphBucketRule)
} derive(Debug, Eq)

///|
pub extend ExecutionGraphBucketError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend ExecutionGraphBucketError with Eq::{not_equal, equal}

///|
#valtype
pub struct ExecutionGraphShape {
  priv phase : ExecutionGraphPhase
  priv batch_rows : Int32
  priv query_tokens : Int32
  priv context_tokens : Int32
}

///|
#valtype
pub struct ExecutionGraphBucket {
  priv slot : Int32
  priv phase : ExecutionGraphPhase
  priv batch_rows : Int32
  priv query_tokens : Int32
  priv context_tokens : Int32
}

///|
/// Compact startup table metadata; backends own captured resources by slot.
pub struct ExecutionGraphBucketTable {
  priv maximum_batch_exponent : Int32
  priv maximum_query_exponent : Int32
  priv maximum_context_exponent : Int32
  priv phase_stride : Int32
  priv query_stride : Int32
  priv context_stride : Int32
  priv bucket_count : Int32
}
