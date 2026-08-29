///|
pub struct TensorParallelTopology {
  priv rank : Int
  priv world_size : Int
} derive(Debug, Eq)

///|
pub(all) enum TensorPlacement {
  Replicated
  ShardRows
  ShardColumns
} derive(Debug, Eq)

///|
pub struct TensorShardExtent {
  priv full_rows : Int64
  priv full_columns : Int64
  priv row_start : Int64
  priv row_count : Int64
  priv column_start : Int64
  priv column_count : Int64
  priv byte_length : Int64
} derive(Debug, Eq)

///|
pub struct TensorArenaRegion {
  priv offset_bytes : Int64
  priv byte_length : Int64
  priv alignment_bytes : Int64
} derive(Debug, Eq)

///|
pub struct TensorTransferSegment {
  priv source_payload_offset_bytes : Int64
  priv destination_offset_bytes : Int64
  priv byte_length : Int64
} derive(Debug, Eq)

///|
pub struct TensorTransferRecipe {
  priv reference : @plan.TensorRef
  priv source_base_bytes : Int64
  priv source_stride_bytes : Int64
  priv destination_base_bytes : Int64
  priv destination_stride_bytes : Int64
  priv segment_bytes : Int64
  priv segment_count : Int64
} derive(Debug, Eq)

///|
pub struct TensorRankPlacement {
  priv reference : @plan.TensorRef
  priv placement : TensorPlacement
  priv extent : TensorShardExtent
  priv region : TensorArenaRegion
  priv recipe : TensorTransferRecipe
} derive(Debug, Eq)

///|
/// Exact rank-local launch geometry for one semantic operation. Model-family
/// builders choose the widths; generic consumers bind them by stable operation
/// identity without reconstructing a family graph.
pub struct TensorParallelOperationPlan {
  priv operation : @plan.OperationId
  priv kind : @plan.OperationKind
  priv capability : @plan.KernelCapabilityId
  priv input_width : Int
  priv output_width : Int
  priv inner_width : Int
  priv query_head_count : Int
  priv kv_head_count : Int
  priv head_dimension : Int
} derive(Debug, Eq)

///|
pub struct TensorCollectiveSequenceId(Int) derive(Debug, Eq)

///|
pub(all) enum TensorCollectiveKind {
  SumAllReduce
  LastAxisAllGather
} derive(Debug, Eq)

///|
pub struct TensorCollectiveSite {
  priv sequence : TensorCollectiveSequenceId
  priv operation : @plan.OperationId
  priv kind : TensorCollectiveKind
  priv full_width : Int
  priv local_width : Int
} derive(Debug, Eq)

///|
pub struct TensorParallelRankPlan {
  priv identity : @spec.ModelIdentity
  priv topology : TensorParallelTopology
  priv alignment_bytes : Int64
  priv arena_bytes : Int64
  priv materialized_bytes : Int64
  priv tensors : ReadOnlyArray[TensorRankPlacement]
  priv operations : ReadOnlyArray[TensorParallelOperationPlan]
  priv collectives : ReadOnlyArray[TensorCollectiveSite]
} derive(Debug, Eq)
