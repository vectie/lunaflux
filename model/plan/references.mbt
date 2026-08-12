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
