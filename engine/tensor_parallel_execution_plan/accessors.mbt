///|
pub fn TensorParallelExecutionPlanDigest::as_sha256(
  self : TensorParallelExecutionPlanDigest,
) -> String {
  self.sha256
}

///|
pub fn TensorParallelScalarRegion::space(
  self : TensorParallelScalarRegion,
) -> TensorParallelMemorySpace {
  self.space
}

///|
pub fn TensorParallelScalarRegion::offset_bytes(
  self : TensorParallelScalarRegion,
) -> Int64 {
  self.offset_bytes
}

///|
pub fn TensorParallelScalarRegion::byte_count(
  self : TensorParallelScalarRegion,
) -> Int64 {
  self.byte_count
}

///|
pub fn TensorParallelScalarRegion::reserved_bytes(
  self : TensorParallelScalarRegion,
) -> Int64 {
  self.reserved_bytes
}

///|
pub fn TensorParallelScalarRegion::alignment(
  self : TensorParallelScalarRegion,
) -> Int64 {
  self.alignment
}

///|
pub fn TensorParallelScalarRegion::end_offset_bytes(
  self : TensorParallelScalarRegion,
) -> Int64 {
  self.offset_bytes + self.reserved_bytes
}

///|
pub fn TensorParallelDescriptorBinding::role(
  self : TensorParallelDescriptorBinding,
) -> TensorParallelDescriptorRole {
  self.role
}

///|
pub fn TensorParallelDescriptorBinding::region(
  self : TensorParallelDescriptorBinding,
) -> TensorParallelScalarRegion {
  self.region
}

///|
pub fn TensorParallelActivationBinding::reference(
  self : TensorParallelActivationBinding,
) -> @plan.ActivationRef {
  self.reference
}

///|
pub fn TensorParallelActivationBinding::slot(
  self : TensorParallelActivationBinding,
) -> Int {
  self.slot
}

///|
pub fn TensorParallelActivationBinding::produced_by(
  self : TensorParallelActivationBinding,
) -> @plan.OperationId {
  self.produced_by
}

///|
pub fn TensorParallelActivationBinding::live_through_step(
  self : TensorParallelActivationBinding,
) -> Int {
  self.live_through_step
}

///|
pub fn TensorParallelActivationBinding::region(
  self : TensorParallelActivationBinding,
) -> TensorParallelScalarRegion {
  self.region
}

///|
pub fn TensorParallelPhysicalOperand::role(
  self : TensorParallelPhysicalOperand,
) -> @tensor_parallel_launch_contract.TensorParallelOperandRole {
  self.role
}

///|
pub fn TensorParallelPhysicalOperand::region(
  self : TensorParallelPhysicalOperand,
) -> TensorParallelScalarRegion {
  self.region
}

///|
pub fn TensorParallelOperationBinding::operation_id(
  self : TensorParallelOperationBinding,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn TensorParallelOperationBinding::operation_kind(
  self : TensorParallelOperationBinding,
) -> @plan.OperationKind {
  self.operation_kind
}

///|
pub fn TensorParallelOperationBinding::capability(
  self : TensorParallelOperationBinding,
) -> @plan.KernelCapabilityId {
  self.capability
}

///|
pub fn TensorParallelOperationBinding::entry_point(
  self : TensorParallelOperationBinding,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn TensorParallelOperationBinding::module_digest(
  self : TensorParallelOperationBinding,
) -> @catalog.AotArtifactDigest {
  self.module_digest
}

///|
pub fn TensorParallelOperationBinding::function_symbol(
  self : TensorParallelOperationBinding,
) -> String {
  self.function_symbol
}

///|
pub fn TensorParallelOperationBinding::dimensions(
  self : TensorParallelOperationBinding,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn TensorParallelOperationBinding::operands(
  self : TensorParallelOperationBinding,
) -> ReadOnlyArray[TensorParallelPhysicalOperand] {
  self.operands
}

///|
pub fn TensorParallelCollectiveBinding::sequence(
  self : TensorParallelCollectiveBinding,
) -> Int {
  self.sequence
}

///|
pub fn TensorParallelCollectiveBinding::operation_id(
  self : TensorParallelCollectiveBinding,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn TensorParallelCollectiveBinding::kind(
  self : TensorParallelCollectiveBinding,
) -> TensorParallelPhysicalCollectiveKind {
  self.kind
}

///|
pub fn TensorParallelCollectiveBinding::region(
  self : TensorParallelCollectiveBinding,
) -> TensorParallelScalarRegion {
  self.region
}

///|
pub fn TensorParallelCollectiveBinding::local_send_offset_bytes(
  self : TensorParallelCollectiveBinding,
) -> Int64 {
  self.local_send_offset_bytes
}

///|
pub fn TensorParallelCollectiveBinding::local_send_bytes(
  self : TensorParallelCollectiveBinding,
) -> Int64 {
  self.local_send_bytes
}

///|
pub fn TensorParallelCollectiveBinding::receive_bytes(
  self : TensorParallelCollectiveBinding,
) -> Int64 {
  self.receive_bytes
}

///|
pub fn TensorParallelExecutionPlan::identity(
  self : TensorParallelExecutionPlan,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn TensorParallelExecutionPlan::rank(
  self : TensorParallelExecutionPlan,
) -> Int {
  self.rank
}

///|
pub fn TensorParallelExecutionPlan::world_size(
  self : TensorParallelExecutionPlan,
) -> Int {
  self.world_size
}

///|
pub fn TensorParallelExecutionPlan::device_ordinal(
  self : TensorParallelExecutionPlan,
) -> Int {
  self.device_ordinal
}

///|
pub fn TensorParallelExecutionPlan::target(
  self : TensorParallelExecutionPlan,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn TensorParallelExecutionPlan::catalog_version(
  self : TensorParallelExecutionPlan,
) -> @catalog.CatalogVersion {
  self.catalog_version
}

///|
pub fn TensorParallelExecutionPlan::profile(
  self : TensorParallelExecutionPlan,
) -> @launch_contract.PagedKernelProfile {
  self.profile
}

///|
pub fn TensorParallelExecutionPlan::digest(
  self : TensorParallelExecutionPlan,
) -> TensorParallelExecutionPlanDigest {
  self.digest
}

///|
pub fn TensorParallelExecutionPlan::descriptor_bytes(
  self : TensorParallelExecutionPlan,
) -> Int64 {
  self.descriptor_bytes
}

///|
pub fn TensorParallelExecutionPlan::rank_constants(
  self : TensorParallelExecutionPlan,
) -> TensorParallelScalarRegion {
  self.rank_constants
}

///|
pub fn TensorParallelExecutionPlan::weight_arena_bytes(
  self : TensorParallelExecutionPlan,
) -> Int64 {
  self.weight_arena_bytes
}

///|
pub fn TensorParallelExecutionPlan::kv_arena_bytes(
  self : TensorParallelExecutionPlan,
) -> Int64 {
  self.kv_arena_bytes
}

///|
pub fn TensorParallelExecutionPlan::activation_workspace_bytes(
  self : TensorParallelExecutionPlan,
) -> Int64 {
  self.activation_workspace_bytes
}

///|
pub fn TensorParallelExecutionPlan::workspace(
  self : TensorParallelExecutionPlan,
) -> TensorParallelScalarRegion {
  self.workspace
}

///|
pub fn TensorParallelExecutionPlan::descriptors(
  self : TensorParallelExecutionPlan,
) -> ReadOnlyArray[TensorParallelDescriptorBinding] {
  self.descriptors
}

///|
pub fn TensorParallelExecutionPlan::activation(
  self : TensorParallelExecutionPlan,
  reference : @plan.ActivationRef,
) -> TensorParallelActivationBinding? {
  let index = reference.as_int()
  guard self.activations.get(index) is Some(binding) &&
    binding.reference().as_int() == index else {
    return None
  }
  Some(binding)
}

///|
pub fn TensorParallelExecutionPlan::activation_count(
  self : TensorParallelExecutionPlan,
) -> Int {
  self.activations.length()
}

///|
pub fn TensorParallelExecutionPlan::operation(
  self : TensorParallelExecutionPlan,
  operation : @plan.OperationId,
) -> TensorParallelOperationBinding? {
  let index = operation.as_int()
  guard self.operations.get(index) is Some(binding) &&
    binding.operation_id().as_int() == index else {
    return None
  }
  Some(binding)
}

///|
/// Returns the exact warmed operation binding without constructing an
/// optional success capability. Rejection is confined to the hostile path.
pub fn TensorParallelExecutionPlan::operation_exact(
  self : TensorParallelExecutionPlan,
  operation : @plan.OperationId,
) -> TensorParallelOperationBinding raise TensorParallelExecutionPlanError {
  let index = operation.as_int()
  guard self.operations.get(index) is Some(binding) &&
    binding.operation_id().as_int() == index else {
    raise rejected(OperationOrder, operation=index)
  }
  binding
}

///|
pub fn TensorParallelExecutionPlan::operation_count(
  self : TensorParallelExecutionPlan,
) -> Int {
  self.operations.length()
}

///|
pub fn TensorParallelExecutionPlan::collective(
  self : TensorParallelExecutionPlan,
  index : Int,
) -> TensorParallelCollectiveBinding? {
  self.collectives.get(index)
}

///|
/// Returns the exact warmed collective binding without constructing an
/// optional success capability. Rejection is confined to the hostile path.
pub fn TensorParallelExecutionPlan::collective_exact(
  self : TensorParallelExecutionPlan,
  index : Int,
) -> TensorParallelCollectiveBinding raise TensorParallelExecutionPlanError {
  guard self.collectives.get(index) is Some(binding) &&
    binding.sequence() == index + 1 else {
    raise rejected(CollectiveSequence, operation=index)
  }
  binding
}

///|
pub fn TensorParallelExecutionPlan::collective_count(
  self : TensorParallelExecutionPlan,
) -> Int {
  self.collectives.length()
}
