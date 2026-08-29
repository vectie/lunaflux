///|
const MIN_TENSOR_PARALLEL_RANKS : Int = 2

///|
const MAX_TENSOR_PARALLEL_RANKS : Int = 16

///|
const MIN_RANK_MEMORY_CEILING_BYTES : Int64 = 268435456L

///|
const MAX_RANK_MEMORY_CEILING_BYTES : Int64 = 1125899906842624L

///|
const MAX_INT64 : Int64 = 0x7fffffffffffffffL

///|
/// Explicit rank identity and memory claim. It owns no device or allocation.
pub struct TensorParallelRankDeclaration {
  priv rank : Int
  priv world_size : Int
  priv process_ordinal : Int
  priv device_name : String
  priv target : @catalog.DeviceTarget
  priv peer_rank_count : Int
  priv memory_ceiling_bytes : Int64
  priv weight_reserved_bytes : Int64
  priv activation_workspace_reserved_bytes : Int64
  priv kv_reserved_bytes : Int64
  priv collective_reserved_bytes : Int64
  priv reserved_total_bytes : Int64
} derive(Debug, Eq)

///|
/// Explicit deployment declaration. Unsupported host and pipeline counts are
/// representable so startup admission can reject them deterministically.
pub struct TensorParallelRuntimeDeclaration {
  priv host_count : Int
  priv pipeline_stage_count : Int
  priv ranks : ReadOnlyArray[TensorParallelRankDeclaration]
} derive(Debug, Eq)

///|
/// Exact admitted capacity and identity for one stable rank.
pub struct TensorParallelRankCapacity {
  priv rank : Int
  priv world_size : Int
  priv process_ordinal : Int
  priv device_name : String
  priv target : @catalog.DeviceTarget
  priv peer_rank_count : Int
  priv physical_bytes : Int64
  priv memory_ceiling_bytes : Int64
  priv weight_reserved_bytes : Int64
  priv activation_workspace_reserved_bytes : Int64
  priv kv_reserved_bytes : Int64
  priv collective_reserved_bytes : Int64
  priv reserved_total_bytes : Int64
  priv available_bytes : Int64
} derive(Debug, Eq)

///|
/// Checked aggregate of the exact admitted per-rank capacities.
pub struct TensorParallelAggregateCapacity {
  priv physical_bytes : Int64
  priv memory_ceiling_bytes : Int64
  priv weight_reserved_bytes : Int64
  priv activation_workspace_reserved_bytes : Int64
  priv kv_reserved_bytes : Int64
  priv collective_reserved_bytes : Int64
  priv reserved_total_bytes : Int64
  priv available_bytes : Int64
  priv minimum_rank_available_bytes : Int64
} derive(Debug, Eq)

///|
/// Root-free, owner-free local tensor-parallel startup foundation.
pub struct TensorParallelRuntimeAdmission {
  priv driver_version : Int
  priv world_size : Int
  priv target : @catalog.DeviceTarget
  priv peer_link_count : Int
  priv ranks : ReadOnlyArray[TensorParallelRankCapacity]
  priv aggregate : TensorParallelAggregateCapacity
} derive(Debug, Eq)

///|
priv struct TopologyRankFacts {
  rank : Int
  process_ordinal : Int
  device_name : String
  physical_bytes : Int64
  target : @catalog.DeviceTarget
}

///|
priv struct TopologyFacts {
  driver_version : Int
  world_size : Int
  target : @catalog.DeviceTarget
  peer_link_count : Int
  ranks : ReadOnlyArray[TopologyRankFacts]
}
