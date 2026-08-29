///|
pub fn TensorParallelRuntimeAdmission::driver_version(
  self : TensorParallelRuntimeAdmission,
) -> Int {
  self.driver_version
}

///|
pub fn TensorParallelRuntimeAdmission::world_size(
  self : TensorParallelRuntimeAdmission,
) -> Int {
  self.world_size
}

///|
pub fn TensorParallelRuntimeAdmission::target(
  self : TensorParallelRuntimeAdmission,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn TensorParallelRuntimeAdmission::peer_link_count(
  self : TensorParallelRuntimeAdmission,
) -> Int {
  self.peer_link_count
}

///|
pub fn TensorParallelRuntimeAdmission::rank_capacity(
  self : TensorParallelRuntimeAdmission,
  rank : Int,
) -> TensorParallelRankCapacity? {
  self.ranks.get(rank)
}

///|
pub fn TensorParallelRuntimeAdmission::rank_capacities(
  self : TensorParallelRuntimeAdmission,
) -> ReadOnlyArray[TensorParallelRankCapacity] {
  self.ranks
}

///|
pub fn TensorParallelRuntimeAdmission::aggregate_capacity(
  self : TensorParallelRuntimeAdmission,
) -> TensorParallelAggregateCapacity {
  self.aggregate
}

///|
pub fn TensorParallelRankCapacity::rank(
  self : TensorParallelRankCapacity,
) -> Int {
  self.rank
}

///|
pub fn TensorParallelRankCapacity::world_size(
  self : TensorParallelRankCapacity,
) -> Int {
  self.world_size
}

///|
pub fn TensorParallelRankCapacity::process_ordinal(
  self : TensorParallelRankCapacity,
) -> Int {
  self.process_ordinal
}

///|
pub fn TensorParallelRankCapacity::device_name(
  self : TensorParallelRankCapacity,
) -> String {
  self.device_name
}

///|
pub fn TensorParallelRankCapacity::target(
  self : TensorParallelRankCapacity,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn TensorParallelRankCapacity::peer_rank_count(
  self : TensorParallelRankCapacity,
) -> Int {
  self.peer_rank_count
}

///|
pub fn TensorParallelRankCapacity::physical_bytes(
  self : TensorParallelRankCapacity,
) -> Int64 {
  self.physical_bytes
}

///|
pub fn TensorParallelRankCapacity::memory_ceiling_bytes(
  self : TensorParallelRankCapacity,
) -> Int64 {
  self.memory_ceiling_bytes
}

///|
pub fn TensorParallelRankCapacity::weight_reserved_bytes(
  self : TensorParallelRankCapacity,
) -> Int64 {
  self.weight_reserved_bytes
}

///|
pub fn TensorParallelRankCapacity::activation_workspace_reserved_bytes(
  self : TensorParallelRankCapacity,
) -> Int64 {
  self.activation_workspace_reserved_bytes
}

///|
pub fn TensorParallelRankCapacity::kv_reserved_bytes(
  self : TensorParallelRankCapacity,
) -> Int64 {
  self.kv_reserved_bytes
}

///|
pub fn TensorParallelRankCapacity::collective_reserved_bytes(
  self : TensorParallelRankCapacity,
) -> Int64 {
  self.collective_reserved_bytes
}

///|
pub fn TensorParallelRankCapacity::reserved_total_bytes(
  self : TensorParallelRankCapacity,
) -> Int64 {
  self.reserved_total_bytes
}

///|
pub fn TensorParallelRankCapacity::available_bytes(
  self : TensorParallelRankCapacity,
) -> Int64 {
  self.available_bytes
}

///|
pub fn TensorParallelAggregateCapacity::physical_bytes(
  self : TensorParallelAggregateCapacity,
) -> Int64 {
  self.physical_bytes
}

///|
pub fn TensorParallelAggregateCapacity::memory_ceiling_bytes(
  self : TensorParallelAggregateCapacity,
) -> Int64 {
  self.memory_ceiling_bytes
}

///|
pub fn TensorParallelAggregateCapacity::weight_reserved_bytes(
  self : TensorParallelAggregateCapacity,
) -> Int64 {
  self.weight_reserved_bytes
}

///|
pub fn TensorParallelAggregateCapacity::activation_workspace_reserved_bytes(
  self : TensorParallelAggregateCapacity,
) -> Int64 {
  self.activation_workspace_reserved_bytes
}

///|
pub fn TensorParallelAggregateCapacity::kv_reserved_bytes(
  self : TensorParallelAggregateCapacity,
) -> Int64 {
  self.kv_reserved_bytes
}

///|
pub fn TensorParallelAggregateCapacity::collective_reserved_bytes(
  self : TensorParallelAggregateCapacity,
) -> Int64 {
  self.collective_reserved_bytes
}

///|
pub fn TensorParallelAggregateCapacity::reserved_total_bytes(
  self : TensorParallelAggregateCapacity,
) -> Int64 {
  self.reserved_total_bytes
}

///|
pub fn TensorParallelAggregateCapacity::available_bytes(
  self : TensorParallelAggregateCapacity,
) -> Int64 {
  self.available_bytes
}

///|
pub fn TensorParallelAggregateCapacity::minimum_rank_available_bytes(
  self : TensorParallelAggregateCapacity,
) -> Int64 {
  self.minimum_rank_available_bytes
}
