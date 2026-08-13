///|
/// Stable position of an operation in an ordered model plan.
pub struct OperationId(Int) derive(Debug, Eq)

///|
/// Stable reference to model data materialized by a later loader phase.
///
/// The reference is semantic plan data. It is not a tensor, allocation,
/// address, file offset, or device handle.
pub struct TensorRef(Int) derive(Debug, Eq)

///|
/// Stable reference to an intermediate value produced by an earlier operation.
pub struct ActivationRef(Int) derive(Debug, Eq)

///|
/// Stable semantic decoder-layer identity for persistent KV ownership.
///
/// This is neither an operation position nor a device byte offset. Model-plan
/// validation binds the complete ordered attention graph to the cache geometry
/// so later planners never recover a layer by counting operations.
pub struct KvLayerId(Int) derive(Debug, Eq)

///|
/// Architecture-neutral external input vocabulary for model plans.
pub(all) enum ModelInputRef {
  TokenIds
} derive(Debug, Eq)

///|
/// Value consumed by a semantic operation.
pub(all) enum ValueRef {
  ModelInput(ModelInputRef)
  Tensor(TensorRef)
  Activation(ActivationRef)
} derive(Debug, Eq)

///|
/// Architecture-neutral live inputs required by cached attention execution.
///
/// `StepCounts` is the scalar bundle containing at least the live query-row,
/// query-token, and page-table-entry counts. Those values bound every packed
/// offset and payload read; its concrete element layout remains execution-owned.
///
/// Query and page-table offsets use packed row boundaries. Concrete element
/// widths, device allocations, transfer layouts, and kernel ABI positions are
/// selected by later execution packages.
pub(all) enum OperationRuntimeInput {
  StepCounts
  QueryPositions
  QueryRowOffsets
  SequenceLengths
  PageTableOffsets
  PageTablePageIndices
} derive(Debug, Eq)

///|
/// Explicit persistent-state effect of one semantic operation.
pub(all) enum PersistentStateAccess {
  NoPersistentState
  KvCacheReadWrite(KvLayerId)
} derive(Debug, Eq)

///|
/// Stable integer identity for a semantic kernel capability.
///
/// IDs identify behavior, never a vendor API or a concrete kernel artifact.
pub struct KernelCapabilityId(Int) derive(Debug, Eq)

///|
pub fn OperationId::new(value : Int) -> OperationId raise PlanValidationError {
  if value < 0 {
    raise InvalidIdentifier(Operation)
  }
  OperationId(value)
}

///|
pub fn OperationId::as_int(self : OperationId) -> Int {
  self.0
}

///|
pub fn TensorRef::new(value : Int) -> TensorRef raise PlanValidationError {
  if value < 0 {
    raise InvalidIdentifier(Tensor)
  }
  TensorRef(value)
}

///|
pub fn TensorRef::as_int(self : TensorRef) -> Int {
  self.0
}

///|
pub fn ActivationRef::new(
  value : Int,
) -> ActivationRef raise PlanValidationError {
  if value < 0 {
    raise InvalidIdentifier(Activation)
  }
  ActivationRef(value)
}

///|
pub fn ActivationRef::as_int(self : ActivationRef) -> Int {
  self.0
}

///|
pub fn KvLayerId::new(value : Int) -> KvLayerId raise PlanValidationError {
  if value < 0 {
    raise InvalidIdentifier(KvLayer)
  }
  KvLayerId(value)
}

///|
pub fn KvLayerId::as_int(self : KvLayerId) -> Int {
  self.0
}

///|
pub fn KernelCapabilityId::from_stable_int(
  value : Int,
) -> KernelCapabilityId raise PlanValidationError {
  if value <= 0 {
    raise InvalidIdentifier(KernelCapability)
  }
  KernelCapabilityId(value)
}

///|
pub fn KernelCapabilityId::as_int(self : KernelCapabilityId) -> Int {
  self.0
}

///|
/// Semantic capability 1: gather model-width rows from an embedding table.
pub fn KernelCapabilityId::embedding_lookup() -> KernelCapabilityId {
  KernelCapabilityId(1)
}

///|
/// Semantic capability 2: root-mean-square normalization.
pub fn KernelCapabilityId::rms_norm() -> KernelCapabilityId {
  KernelCapabilityId(2)
}

///|
/// Semantic capability 3: query, key, and value projection.
pub fn KernelCapabilityId::qkv_projection() -> KernelCapabilityId {
  KernelCapabilityId(3)
}

///|
/// Semantic capability 4: rotary position transformation.
pub fn KernelCapabilityId::rotary_embedding() -> KernelCapabilityId {
  KernelCapabilityId(4)
}

///|
/// Semantic capability 5: causal grouped-query attention.
pub fn KernelCapabilityId::causal_attention() -> KernelCapabilityId {
  KernelCapabilityId(5)
}

///|
/// Semantic capability 6: dense matrix projection.
pub fn KernelCapabilityId::dense_projection() -> KernelCapabilityId {
  KernelCapabilityId(6)
}

///|
/// Semantic capability 7: elementwise residual addition.
pub fn KernelCapabilityId::residual_add() -> KernelCapabilityId {
  KernelCapabilityId(7)
}

///|
/// Semantic capability 8: gated feed-forward transformation.
pub fn KernelCapabilityId::gated_mlp() -> KernelCapabilityId {
  KernelCapabilityId(8)
}

///|
/// Semantic capability 9: RoPE using explicit per-query positions.
pub fn KernelCapabilityId::positioned_rotary_embedding() -> KernelCapabilityId {
  KernelCapabilityId(9)
}

///|
/// Semantic capability 10: paged causal attention with persistent KV update.
pub fn KernelCapabilityId::paged_causal_attention() -> KernelCapabilityId {
  KernelCapabilityId(10)
}
