///|
pub(all) enum ExecutionGraphPhase {
  PrefillGraph
  DecodeGraph
  MixedGraph
} derive(Debug, Eq)

///|
pub(all) enum ExecutionGraphBucketRule {
  Shape
  Limit
  Capacity
} derive(Debug, Eq)

///|
pub(all) suberror ExecutionGraphBucketError {
  InvalidExecutionGraphBucket(ExecutionGraphBucketRule)
} derive(Debug, Eq)

///|
pub struct ExecutionGraphShape {
  priv phase : ExecutionGraphPhase
  priv batch_rows : Int
  priv query_tokens : Int
  priv context_tokens : Int
}

///|
pub struct ExecutionGraphBucket {
  priv slot : Int
  priv phase : ExecutionGraphPhase
  priv batch_rows : Int
  priv query_tokens : Int
  priv context_tokens : Int
}

///|
/// Compact startup table metadata; backends own captured resources by slot.
pub struct ExecutionGraphBucketTable {
  priv maximum_batch_exponent : Int
  priv maximum_query_exponent : Int
  priv maximum_context_exponent : Int
  priv phase_stride : Int
  priv query_stride : Int
  priv context_stride : Int
  priv bucket_count : Int
}
