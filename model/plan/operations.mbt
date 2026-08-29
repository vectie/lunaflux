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
  priv id : OperationId
  priv kind : OperationKind
  priv inputs : ReadOnlyArray[ValueRef]
  priv outputs : ReadOnlyArray[ActivationRef]
  priv shape : OperationShape
  priv required_capability : KernelCapabilityId
  priv workspace_bytes : Int64
  priv kv_layer : KvLayerId?
  priv runtime_inputs : ReadOnlyArray[OperationRuntimeInput]
  priv persistent_state : PersistentStateAccess
} derive(Debug, Eq)

///|
fn new_operation(
  id : OperationId,
  kind : OperationKind,
  inputs : ArrayView[ValueRef],
  outputs : ArrayView[ActivationRef],
  shape : OperationShape,
  required_capability : KernelCapabilityId,
  workspace_bytes : Int64,
  kv_layer : KvLayerId?,
  runtime_inputs : ArrayView[OperationRuntimeInput],
  persistent_state : PersistentStateAccess,
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
    kv_layer,
    runtime_inputs: ReadOnlyArray::from_array(runtime_inputs),
    persistent_state,
  }
}

///|
/// Constructs an operation that has no semantic KV-layer association, live
/// runtime inputs, or persistent state.
///
/// Rotary and causal-attention operations must use `new_with_execution`, even
/// for full-context execution, because their decoder-layer identity is part of
/// the model semantics.
pub fn PlanOperation::new_stateless(
  id~ : OperationId,
  kind~ : OperationKind,
  inputs~ : ArrayView[ValueRef],
  outputs~ : ArrayView[ActivationRef],
  shape~ : OperationShape,
  required_capability~ : KernelCapabilityId,
  workspace_bytes~ : Int64,
) -> PlanOperation raise PlanValidationError {
  if kind is (RotaryEmbedding | CausalAttention) {
    raise InvalidGraph(KvExecutionContract)
  }
  new_operation(
    id,
    kind,
    inputs,
    outputs,
    shape,
    required_capability,
    workspace_bytes,
    None,
    [],
    NoPersistentState,
  )
}

///|
/// Constructs a non-KV operation with an explicit ordered live-input contract.
///
/// This is used by paged full-graph execution where otherwise stateless
/// operations consume `StepCounts` to bound live rows and tokens. Rotary and
/// causal attention remain layered operations and must use
/// `new_with_execution`; an empty live-input list must use `new_stateless`.
pub fn PlanOperation::new_with_runtime_inputs(
  id~ : OperationId,
  kind~ : OperationKind,
  inputs~ : ArrayView[ValueRef],
  outputs~ : ArrayView[ActivationRef],
  shape~ : OperationShape,
  required_capability~ : KernelCapabilityId,
  workspace_bytes~ : Int64,
  runtime_inputs~ : ArrayView[OperationRuntimeInput],
) -> PlanOperation raise PlanValidationError {
  if kind is (RotaryEmbedding | CausalAttention) || runtime_inputs.is_empty() {
    raise InvalidGraph(KvExecutionContract)
  }
  new_operation(
    id,
    kind,
    inputs,
    outputs,
    shape,
    required_capability,
    workspace_bytes,
    None,
    runtime_inputs,
    NoPersistentState,
  )
}

///|
/// Constructs an operation with an explicit decoder layer, live-input
/// contract, and persistent-state effect.
///
/// `ModelPlan::new_with_kv_execution` validates the only legal combinations: full-context
/// rotary/attention operations are layered but stateless, while paged rotary
/// consumes query positions and paged attention consumes the complete ordered
/// row/page contract and reads/writes that exact layer's KV cache.
pub fn PlanOperation::new_with_execution(
  id~ : OperationId,
  kind~ : OperationKind,
  inputs~ : ArrayView[ValueRef],
  outputs~ : ArrayView[ActivationRef],
  shape~ : OperationShape,
  required_capability~ : KernelCapabilityId,
  workspace_bytes~ : Int64,
  kv_layer~ : KvLayerId,
  runtime_inputs~ : ArrayView[OperationRuntimeInput],
  persistent_state~ : PersistentStateAccess,
) -> PlanOperation raise PlanValidationError {
  new_operation(
    id,
    kind,
    inputs,
    outputs,
    shape,
    required_capability,
    workspace_bytes,
    Some(kv_layer),
    runtime_inputs,
    persistent_state,
  )
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

///|
pub fn PlanOperation::kv_layer(self : PlanOperation) -> KvLayerId? {
  self.kv_layer
}

///|
pub fn PlanOperation::runtime_inputs(
  self : PlanOperation,
) -> ReadOnlyArray[OperationRuntimeInput] {
  self.runtime_inputs
}

///|
pub fn PlanOperation::persistent_state(
  self : PlanOperation,
) -> PersistentStateAccess {
  self.persistent_state
}

///|
/// Returns true only for the validated paged-attention state contract.
fn PlanOperation::uses_paged_kv(self : PlanOperation) -> Bool {
  self.persistent_state is KvCacheReadWrite(_)
}

///|
fn runtime_inputs_equal(
  actual : ReadOnlyArray[OperationRuntimeInput],
  expected : ArrayView[OperationRuntimeInput],
) -> Bool {
  if actual.length() != expected.length() {
    return false
  }
  for index, value in actual {
    if value != expected[index] {
      return false
    }
  }
  true
}

///|
fn PlanOperation::has_runtime_inputs(
  self : PlanOperation,
  expected : ArrayView[OperationRuntimeInput],
) -> Bool {
  runtime_inputs_equal(self.runtime_inputs, expected)
}
