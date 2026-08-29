///|
pub fn TensorParallelRankDevice::single(
  device_ordinal~ : Int,
  target~ : @catalog.DeviceTarget,
) -> TensorParallelRankDevice raise TensorParallelDevicePlanError {
  if device_ordinal < 0 || device_ordinal > 1023 {
    raise invalid(DeviceOrdinal, device_ordinal)
  }
  { rank: 0, world_size: 1, device_ordinal, target }
}

///|
pub fn TensorParallelRankDevice::from_local_topology(
  topology : @device_topology.AdmittedLocalTopology,
  rank~ : Int,
) -> TensorParallelRankDevice raise TensorParallelDevicePlanError {
  if rank < 0 || rank >= topology.world_size() {
    raise invalid(Rank, rank)
  }
  guard topology.rank(rank) is Some(device) else { raise invalid(Rank, rank) }
  if device.target() != topology.target() {
    raise invalid(DeviceTarget, rank)
  }
  {
    rank,
    world_size: topology.world_size(),
    device_ordinal: device.ordinal(),
    target: device.target(),
  }
}

///|
pub fn TensorParallelRankDevice::rank(self : TensorParallelRankDevice) -> Int {
  self.rank
}

///|
pub fn TensorParallelRankDevice::world_size(
  self : TensorParallelRankDevice,
) -> Int {
  self.world_size
}

///|
pub fn TensorParallelRankDevice::device_ordinal(
  self : TensorParallelRankDevice,
) -> Int {
  self.device_ordinal
}

///|
pub fn TensorParallelRankDevice::target(
  self : TensorParallelRankDevice,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn TensorParallelMatrixView::reference(
  self : TensorParallelMatrixView,
) -> @plan.TensorRef {
  self.reference
}

///|
pub fn TensorParallelMatrixView::placement(
  self : TensorParallelMatrixView,
) -> @tensor_parallel_plan.TensorPlacement {
  self.placement
}

///|
pub fn TensorParallelMatrixView::full_rows(
  self : TensorParallelMatrixView,
) -> Int64 {
  self.full_rows
}

///|
pub fn TensorParallelMatrixView::full_columns(
  self : TensorParallelMatrixView,
) -> Int64 {
  self.full_columns
}

///|
pub fn TensorParallelMatrixView::local_row_start(
  self : TensorParallelMatrixView,
) -> Int64 {
  self.local_row_start
}

///|
pub fn TensorParallelMatrixView::local_rows(
  self : TensorParallelMatrixView,
) -> Int64 {
  self.local_rows
}

///|
pub fn TensorParallelMatrixView::local_column_start(
  self : TensorParallelMatrixView,
) -> Int64 {
  self.local_column_start
}

///|
pub fn TensorParallelMatrixView::local_columns(
  self : TensorParallelMatrixView,
) -> Int64 {
  self.local_columns
}

///|
pub fn TensorParallelMatrixView::region_offset_bytes(
  self : TensorParallelMatrixView,
) -> Int64 {
  self.region_offset_bytes
}

///|
pub fn TensorParallelMatrixView::region_byte_length(
  self : TensorParallelMatrixView,
) -> Int64 {
  self.region_byte_length
}

///|
pub fn TensorParallelDevicePlan::identity(
  self : TensorParallelDevicePlan,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn TensorParallelDevicePlan::rank(self : TensorParallelDevicePlan) -> Int {
  self.device.rank
}

///|
pub fn TensorParallelDevicePlan::world_size(
  self : TensorParallelDevicePlan,
) -> Int {
  self.device.world_size
}

///|
pub fn TensorParallelDevicePlan::device_ordinal(
  self : TensorParallelDevicePlan,
) -> Int {
  self.device.device_ordinal
}

///|
pub fn TensorParallelDevicePlan::target(
  self : TensorParallelDevicePlan,
) -> @catalog.DeviceTarget {
  self.device.target
}

///|
pub fn TensorParallelDevicePlan::arena_bytes(
  self : TensorParallelDevicePlan,
) -> Int64 {
  self.arena_bytes
}

///|
pub fn TensorParallelDevicePlan::materialized_bytes(
  self : TensorParallelDevicePlan,
) -> Int64 {
  self.materialized_bytes
}

///|
/// Returns the exact authenticated local matrix region for one semantic
/// tensor reference. The ordered array retained by the device plan is the
/// sole physical-view source for this accessor and all convenience views.
pub fn TensorParallelDevicePlan::tensor(
  self : TensorParallelDevicePlan,
  reference : @plan.TensorRef,
) -> TensorParallelMatrixView? {
  let index = reference.as_int()
  guard self.tensors.get(index) is Some(view) &&
    view.reference().as_int() == index else {
    return None
  }
  Some(view)
}

///|
pub fn TensorParallelDevicePlan::tensor_count(
  self : TensorParallelDevicePlan,
) -> Int {
  self.tensors.length()
}

///|
pub fn TensorParallelDevicePlan::operation_count(
  self : TensorParallelDevicePlan,
) -> Int {
  self.operations.length()
}

///|
pub fn TensorParallelDevicePlan::operation(
  self : TensorParallelDevicePlan,
  operation : @plan.OperationId,
) -> @tensor_parallel_plan.TensorParallelOperationPlan? {
  let index = operation.as_int()
  guard self.operations.get(index) is Some(geometry) &&
    geometry.operation().as_int() == index else {
    return None
  }
  Some(geometry)
}

///|
pub fn TensorParallelDevicePlan::collective_count(
  self : TensorParallelDevicePlan,
) -> Int {
  self.collectives.length()
}

///|
pub fn TensorParallelDevicePlan::collective(
  self : TensorParallelDevicePlan,
  index : Int,
) -> @tensor_parallel_plan.TensorCollectiveSite? {
  self.collectives.get(index)
}
