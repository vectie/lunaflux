///|
/// Architecture-neutral semantic operation vocabulary needed by dense Llama
/// v1. Concrete kernel and device selection happen after plan construction.
pub(all) enum OperationKind {
  Embedding
  RmsNorm
  QkvProjection
  RotaryEmbedding
  CausalAttention
  OutputProjection
  ResidualAdd
  GatedMlp
  LanguageModelHead
} derive(Debug, Eq)

///|
/// One immutable node in an ordered semantic model plan.
pub struct PlanOperation {
  id : OperationId
  kind : OperationKind
  inputs : ReadOnlyArray[ValueRef]
  outputs : ReadOnlyArray[ActivationRef]
  shape : OperationShape
  required_capability : KernelCapabilityId
  workspace_bytes : Int64
} derive(Debug, Eq)

///|
pub fn PlanOperation::new(
  id~ : OperationId,
  kind~ : OperationKind,
  inputs~ : ArrayView[ValueRef],
  outputs~ : ArrayView[ActivationRef],
  shape~ : OperationShape,
  required_capability~ : KernelCapabilityId,
  workspace_bytes~ : Int64,
) -> PlanOperation raise PlanValidationError {
  if inputs.is_empty() {
    raise InvalidGraph(EmptyOperationInputs)
  }
  if outputs.is_empty() {
    raise InvalidGraph(EmptyOperationOutputs)
  }
  if workspace_bytes < 0L {
    raise InvalidWorkspaceBound
  }
  {
    id,
    kind,
    inputs: ReadOnlyArray::from_array(inputs),
    outputs: ReadOnlyArray::from_array(outputs),
    shape,
    required_capability,
    workspace_bytes,
  }
}

///|
pub fn PlanOperation::id(self : PlanOperation) -> OperationId {
  self.id
}

///|
pub fn PlanOperation::kind(self : PlanOperation) -> OperationKind {
  self.kind
}

///|
pub fn PlanOperation::inputs(self : PlanOperation) -> ReadOnlyArray[ValueRef] {
  self.inputs
}

///|
pub fn PlanOperation::outputs(
  self : PlanOperation,
) -> ReadOnlyArray[ActivationRef] {
  self.outputs
}

///|
pub fn PlanOperation::shape(self : PlanOperation) -> OperationShape {
  self.shape
}

///|
pub fn PlanOperation::required_capability(
  self : PlanOperation,
) -> KernelCapabilityId {
  self.required_capability
}

///|
pub fn PlanOperation::workspace_bytes(self : PlanOperation) -> Int64 {
  self.workspace_bytes
}
