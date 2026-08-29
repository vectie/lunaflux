///|
/// Immutable rank-to-device identity. It contains no context, allocation, or
/// collective authority.
pub struct TensorParallelRankDevice {
  priv rank : Int
  priv world_size : Int
  priv device_ordinal : Int
  priv target : @catalog.DeviceTarget
} derive(Debug, Eq)

///|
/// One local BF16 matrix view inside the authenticated rank-local arena.
pub struct TensorParallelMatrixView {
  priv reference : @plan.TensorRef
  priv placement : @tensor_parallel_plan.TensorPlacement
  priv full_rows : Int64
  priv full_columns : Int64
  priv local_row_start : Int64
  priv local_rows : Int64
  priv local_column_start : Int64
  priv local_columns : Int64
  priv region_offset_bytes : Int64
  priv region_byte_length : Int64
} derive(Debug, Eq)

///|
/// Immutable, authority-free per-rank execution view.
pub struct TensorParallelDevicePlan {
  priv identity : @spec.ModelIdentity
  priv device : TensorParallelRankDevice
  priv arena_bytes : Int64
  priv materialized_bytes : Int64
  priv tensors : ReadOnlyArray[TensorParallelMatrixView]
  priv operations : ReadOnlyArray[
    @tensor_parallel_plan.TensorParallelOperationPlan,
  ]
  priv collectives : ReadOnlyArray[@tensor_parallel_plan.TensorCollectiveSite]
} derive(Debug, Eq)
