///|
/// Narrow immutable bounds used to plan a BF16 activation/workspace arena.
///
/// `max_arena_bytes` is an exact startup ceiling over activation slots,
/// alignment padding, and the shared kernel workspace. The token-row envelope
/// is derived from the static plan's checked shape constraints.
pub struct DeviceMemoryLimits {
  activation_alignment : Int64
  max_arena_bytes : Int64
} derive(Debug, Eq)

///|
/// Stable identifier for a reusable physical activation slot origin.
pub struct ActivationSlotId(Int) derive(Debug, Eq)

///|
/// Immutable placement and liveness of one semantic activation value.
pub struct ActivationRegion {
  reference : @plan.ActivationRef
  slot : ActivationSlotId
  offset_bytes : Int64
  logical_bytes : Int64
  reserved_bytes : Int64
  produced_by : @plan.OperationId
  live_through_step : Int
  retained_output : Bool
} derive(Debug, Eq)

///|
/// One aligned workspace region reused by every operation in sequence.
pub struct WorkspaceRegion {
  offset_bytes : Int64
  byte_length : Int64
  alignment_bytes : Int64
} derive(Debug, Eq)

///|
/// Pre-resolved activation inputs and outputs for one static operation.
pub struct OperationMemoryBinding {
  operation_id : @plan.OperationId
  activation_inputs : ReadOnlyArray[ActivationRegion]
  activation_outputs : ReadOnlyArray[ActivationRegion]
  workspace : WorkspaceRegion
} derive(Debug, Eq)

///|
/// Deterministic immutable memory plan for one static device plan.
///
/// Repeated token steps consume only these prebuilt regions and bindings; the
/// plan itself performs no allocation and owns no native resource.
pub struct DeviceMemoryPlan {
  identity : @spec.ModelIdentity
  target : @catalog.DeviceTarget
  limits : DeviceMemoryLimits
  token_rows : Int64
  activation_bytes : Int64
  workspace : WorkspaceRegion
  arena_bytes : Int64
  slot_count : Int
  regions : ReadOnlyArray[ActivationRegion]
  bindings : ReadOnlyArray[OperationMemoryBinding]
  final_output : @plan.ActivationRef
} derive(Debug, Eq)

///|
/// Explicit lifecycle owner of the single device activation/workspace arena.
///
/// The allocation alias is intended only for dispatch. Lifecycle authority
/// should remain here, and `close` must run before the context is closed.
pub struct DeviceMemoryArena {
  priv allocation : @device.Allocation
  plan : DeviceMemoryPlan
}

///|
pub fn DeviceMemoryLimits::new(
  activation_alignment~ : Int64,
  max_arena_bytes~ : Int64,
) -> DeviceMemoryLimits raise DeviceMemoryError {
  if activation_alignment < 16L ||
    activation_alignment > 65536L ||
    (activation_alignment & (activation_alignment - 1L)) != 0L {
    raise InvalidLimit(ActivationAlignment)
  }
  if max_arena_bytes <= 0L {
    raise InvalidLimit(ArenaBytes)
  }
  { activation_alignment, max_arena_bytes }
}

///|
pub fn DeviceMemoryLimits::activation_alignment(
  self : DeviceMemoryLimits,
) -> Int64 {
  self.activation_alignment
}

///|
pub fn DeviceMemoryLimits::max_arena_bytes(self : DeviceMemoryLimits) -> Int64 {
  self.max_arena_bytes
}

///|
pub fn ActivationSlotId::as_int(self : ActivationSlotId) -> Int {
  self.0
}

///|
pub fn ActivationRegion::reference(
  self : ActivationRegion,
) -> @plan.ActivationRef {
  self.reference
}

///|
pub fn ActivationRegion::slot(self : ActivationRegion) -> ActivationSlotId {
  self.slot
}

///|
pub fn ActivationRegion::offset_bytes(self : ActivationRegion) -> Int64 {
  self.offset_bytes
}

///|
pub fn ActivationRegion::logical_bytes(self : ActivationRegion) -> Int64 {
  self.logical_bytes
}

///|
pub fn ActivationRegion::reserved_bytes(self : ActivationRegion) -> Int64 {
  self.reserved_bytes
}

///|
pub fn ActivationRegion::end_offset_bytes(self : ActivationRegion) -> Int64 {
  self.offset_bytes + self.reserved_bytes
}

///|
pub fn ActivationRegion::produced_by(
  self : ActivationRegion,
) -> @plan.OperationId {
  self.produced_by
}

///|
pub fn ActivationRegion::live_through_step(self : ActivationRegion) -> Int {
  self.live_through_step
}

///|
pub fn ActivationRegion::is_retained_output(self : ActivationRegion) -> Bool {
  self.retained_output
}

///|
pub fn WorkspaceRegion::offset_bytes(self : WorkspaceRegion) -> Int64 {
  self.offset_bytes
}

///|
pub fn WorkspaceRegion::byte_length(self : WorkspaceRegion) -> Int64 {
  self.byte_length
}

///|
pub fn WorkspaceRegion::alignment_bytes(self : WorkspaceRegion) -> Int64 {
  self.alignment_bytes
}

///|
pub fn WorkspaceRegion::end_offset_bytes(self : WorkspaceRegion) -> Int64 {
  self.offset_bytes + self.byte_length
}

///|
pub fn OperationMemoryBinding::operation_id(
  self : OperationMemoryBinding,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn OperationMemoryBinding::activation_inputs(
  self : OperationMemoryBinding,
) -> ReadOnlyArray[ActivationRegion] {
  self.activation_inputs
}

///|
pub fn OperationMemoryBinding::activation_outputs(
  self : OperationMemoryBinding,
) -> ReadOnlyArray[ActivationRegion] {
  self.activation_outputs
}

///|
pub fn OperationMemoryBinding::workspace(
  self : OperationMemoryBinding,
) -> WorkspaceRegion {
  self.workspace
}

///|
pub fn DeviceMemoryPlan::identity(
  self : DeviceMemoryPlan,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn DeviceMemoryPlan::target(
  self : DeviceMemoryPlan,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn DeviceMemoryPlan::limits(self : DeviceMemoryPlan) -> DeviceMemoryLimits {
  self.limits
}

///|
pub fn DeviceMemoryPlan::token_rows(self : DeviceMemoryPlan) -> Int64 {
  self.token_rows
}

///|
pub fn DeviceMemoryPlan::activation_bytes(self : DeviceMemoryPlan) -> Int64 {
  self.activation_bytes
}

///|
pub fn DeviceMemoryPlan::workspace(self : DeviceMemoryPlan) -> WorkspaceRegion {
  self.workspace
}

///|
pub fn DeviceMemoryPlan::arena_bytes(self : DeviceMemoryPlan) -> Int64 {
  self.arena_bytes
}

///|
pub fn DeviceMemoryPlan::slot_count(self : DeviceMemoryPlan) -> Int {
  self.slot_count
}

///|
pub fn DeviceMemoryPlan::activation_count(self : DeviceMemoryPlan) -> Int {
  self.regions.length()
}

///|
pub fn DeviceMemoryPlan::operation_count(self : DeviceMemoryPlan) -> Int {
  self.bindings.length()
}

///|
pub fn DeviceMemoryPlan::region(
  self : DeviceMemoryPlan,
  reference : @plan.ActivationRef,
) -> ActivationRegion? {
  self.regions.get(reference.as_int())
}

///|
pub fn DeviceMemoryPlan::operation_binding(
  self : DeviceMemoryPlan,
  operation : @plan.OperationId,
) -> OperationMemoryBinding? {
  self.bindings.get(operation.as_int())
}

///|
pub fn DeviceMemoryPlan::final_output(
  self : DeviceMemoryPlan,
) -> @plan.ActivationRef {
  self.final_output
}

///|
pub fn DeviceMemoryArena::plan(self : DeviceMemoryArena) -> DeviceMemoryPlan {
  self.plan
}

///|
pub fn DeviceMemoryArena::identity(
  self : DeviceMemoryArena,
) -> @spec.ModelIdentity {
  self.plan.identity()
}

///|
pub fn DeviceMemoryArena::target(
  self : DeviceMemoryArena,
) -> @catalog.DeviceTarget {
  self.plan.target()
}

///|
/// Returns an alias of the safe opaque allocation for dispatch.
///
/// Closing this alias also closes the arena represented by
/// `DeviceMemoryArena`; callers should retain lifecycle authority here.
pub fn DeviceMemoryArena::allocation(
  self : DeviceMemoryArena,
) -> @device.Allocation {
  self.allocation
}

///|
pub fn DeviceMemoryArena::close(
  self : DeviceMemoryArena,
) -> Unit raise DeviceMemoryError {
  self.allocation.close() catch {
    error => raise DeviceFailed(error)
  }
}
