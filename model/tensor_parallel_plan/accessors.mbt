///|
pub fn TensorParallelTopology::rank(self : TensorParallelTopology) -> Int {
  self.rank
}

///|
pub fn TensorParallelTopology::world_size(self : TensorParallelTopology) -> Int {
  self.world_size
}

///|
pub fn TensorShardExtent::full_rows(self : TensorShardExtent) -> Int64 {
  self.full_rows
}

///|
pub fn TensorShardExtent::full_columns(self : TensorShardExtent) -> Int64 {
  self.full_columns
}

///|
pub fn TensorShardExtent::row_start(self : TensorShardExtent) -> Int64 {
  self.row_start
}

///|
pub fn TensorShardExtent::row_count(self : TensorShardExtent) -> Int64 {
  self.row_count
}

///|
pub fn TensorShardExtent::column_start(self : TensorShardExtent) -> Int64 {
  self.column_start
}

///|
pub fn TensorShardExtent::column_count(self : TensorShardExtent) -> Int64 {
  self.column_count
}

///|
pub fn TensorShardExtent::byte_length(self : TensorShardExtent) -> Int64 {
  self.byte_length
}

///|
pub fn TensorArenaRegion::offset_bytes(self : TensorArenaRegion) -> Int64 {
  self.offset_bytes
}

///|
pub fn TensorArenaRegion::byte_length(self : TensorArenaRegion) -> Int64 {
  self.byte_length
}

///|
pub fn TensorArenaRegion::alignment_bytes(self : TensorArenaRegion) -> Int64 {
  self.alignment_bytes
}

///|
pub fn TensorArenaRegion::end_offset_bytes(self : TensorArenaRegion) -> Int64 {
  self.offset_bytes + self.byte_length
}

///|
pub fn TensorTransferSegment::source_payload_offset_bytes(
  self : TensorTransferSegment,
) -> Int64 {
  self.source_payload_offset_bytes
}

///|
pub fn TensorTransferSegment::destination_offset_bytes(
  self : TensorTransferSegment,
) -> Int64 {
  self.destination_offset_bytes
}

///|
pub fn TensorTransferSegment::byte_length(
  self : TensorTransferSegment,
) -> Int64 {
  self.byte_length
}

///|
pub fn TensorTransferRecipe::segment_count(
  self : TensorTransferRecipe,
) -> Int64 {
  self.segment_count
}

///|
fn TensorTransferRecipe::require_segment(
  self : TensorTransferRecipe,
  index : Int64,
) -> Unit raise TensorParallelPlanError {
  if index < 0L || index >= self.segment_count {
    raise invalid(TransferSegment, index=self.reference.as_int())
  }
}

///|
pub fn TensorTransferRecipe::source_payload_offset_bytes_at(
  self : TensorTransferRecipe,
  index : Int64,
) -> Int64 raise TensorParallelPlanError {
  self.require_segment(index)
  self.source_base_bytes + index * self.source_stride_bytes
}

///|
pub fn TensorTransferRecipe::destination_offset_bytes_at(
  self : TensorTransferRecipe,
  index : Int64,
) -> Int64 raise TensorParallelPlanError {
  self.require_segment(index)
  self.destination_base_bytes + index * self.destination_stride_bytes
}

///|
pub fn TensorTransferRecipe::segment_byte_length(
  self : TensorTransferRecipe,
) -> Int64 {
  self.segment_bytes
}

///|
pub fn TensorTransferRecipe::segment(
  self : TensorTransferRecipe,
  index : Int64,
) -> TensorTransferSegment raise TensorParallelPlanError {
  {
    source_payload_offset_bytes: self.source_payload_offset_bytes_at(index),
    destination_offset_bytes: self.destination_offset_bytes_at(index),
    byte_length: self.segment_bytes,
  }
}

///|
pub fn TensorRankPlacement::reference(
  self : TensorRankPlacement,
) -> @plan.TensorRef {
  self.reference
}

///|
pub fn TensorRankPlacement::placement(
  self : TensorRankPlacement,
) -> TensorPlacement {
  self.placement
}

///|
pub fn TensorRankPlacement::extent(
  self : TensorRankPlacement,
) -> TensorShardExtent {
  self.extent
}

///|
pub fn TensorRankPlacement::region(
  self : TensorRankPlacement,
) -> TensorArenaRegion {
  self.region
}

///|
pub fn TensorRankPlacement::transfer_recipe(
  self : TensorRankPlacement,
) -> TensorTransferRecipe {
  self.recipe
}

///|
pub fn TensorParallelOperationPlan::operation(
  self : TensorParallelOperationPlan,
) -> @plan.OperationId {
  self.operation
}

///|
pub fn TensorParallelOperationPlan::kind(
  self : TensorParallelOperationPlan,
) -> @plan.OperationKind {
  self.kind
}

///|
pub fn TensorParallelOperationPlan::capability(
  self : TensorParallelOperationPlan,
) -> @plan.KernelCapabilityId {
  self.capability
}

///|
pub fn TensorParallelOperationPlan::input_width(
  self : TensorParallelOperationPlan,
) -> Int {
  self.input_width
}

///|
pub fn TensorParallelOperationPlan::output_width(
  self : TensorParallelOperationPlan,
) -> Int {
  self.output_width
}

///|
pub fn TensorParallelOperationPlan::inner_width(
  self : TensorParallelOperationPlan,
) -> Int {
  self.inner_width
}

///|
pub fn TensorParallelOperationPlan::query_head_count(
  self : TensorParallelOperationPlan,
) -> Int {
  self.query_head_count
}

///|
pub fn TensorParallelOperationPlan::kv_head_count(
  self : TensorParallelOperationPlan,
) -> Int {
  self.kv_head_count
}

///|
pub fn TensorParallelOperationPlan::head_dimension(
  self : TensorParallelOperationPlan,
) -> Int {
  self.head_dimension
}

///|
pub fn TensorCollectiveSequenceId::as_int(
  self : TensorCollectiveSequenceId,
) -> Int {
  self.0
}

///|
pub fn TensorCollectiveSite::sequence(
  self : TensorCollectiveSite,
) -> TensorCollectiveSequenceId {
  self.sequence
}

///|
pub fn TensorCollectiveSite::operation(
  self : TensorCollectiveSite,
) -> @plan.OperationId {
  self.operation
}

///|
pub fn TensorCollectiveSite::kind(
  self : TensorCollectiveSite,
) -> TensorCollectiveKind {
  self.kind
}

///|
pub fn TensorCollectiveSite::full_width(self : TensorCollectiveSite) -> Int {
  self.full_width
}

///|
pub fn TensorCollectiveSite::local_width(self : TensorCollectiveSite) -> Int {
  self.local_width
}

///|
pub fn TensorParallelRankPlan::identity(
  self : TensorParallelRankPlan,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn TensorParallelRankPlan::topology(
  self : TensorParallelRankPlan,
) -> TensorParallelTopology {
  self.topology
}

///|
pub fn TensorParallelRankPlan::alignment_bytes(
  self : TensorParallelRankPlan,
) -> Int64 {
  self.alignment_bytes
}

///|
pub fn TensorParallelRankPlan::arena_bytes(
  self : TensorParallelRankPlan,
) -> Int64 {
  self.arena_bytes
}

///|
pub fn TensorParallelRankPlan::materialized_bytes(
  self : TensorParallelRankPlan,
) -> Int64 {
  self.materialized_bytes
}

///|
pub fn TensorParallelRankPlan::tensor_count(
  self : TensorParallelRankPlan,
) -> Int {
  self.tensors.length()
}

///|
pub fn TensorParallelRankPlan::tensor(
  self : TensorParallelRankPlan,
  reference : @plan.TensorRef,
) -> TensorRankPlacement? {
  let index = reference.as_int()
  guard self.tensors.get(index) is Some(value) &&
    value.reference.as_int() == index else {
    return None
  }
  Some(value)
}

///|
pub fn TensorParallelRankPlan::operation_count(
  self : TensorParallelRankPlan,
) -> Int {
  self.operations.length()
}

///|
pub fn TensorParallelRankPlan::operation(
  self : TensorParallelRankPlan,
  operation : @plan.OperationId,
) -> TensorParallelOperationPlan? {
  let index = operation.as_int()
  guard self.operations.get(index) is Some(value) &&
    value.operation.as_int() == index else {
    return None
  }
  Some(value)
}

///|
pub fn TensorParallelRankPlan::collective_count(
  self : TensorParallelRankPlan,
) -> Int {
  self.collectives.length()
}

///|
pub fn TensorParallelRankPlan::collective(
  self : TensorParallelRankPlan,
  index : Int,
) -> TensorCollectiveSite? {
  self.collectives.get(index)
}
