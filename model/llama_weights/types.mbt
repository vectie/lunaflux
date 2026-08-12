///|
/// Stable reasons a validated safetensors manifest cannot bind to a dense
/// Llama plan.
pub(all) suberror LlamaWeightsError {
  TensorCountMismatch(expected~ : Int, actual~ : Int)
  MissingTensor(reference~ : Int)
  ShapeRankMismatch(reference~ : Int)
  ShapeDimensionMismatch(reference~ : Int, axis~ : Int)
} derive(Debug, Eq)

///|
/// One immutable mapping from a semantic model-plan tensor reference to
/// validated safetensors metadata.
pub struct TensorBinding {
  reference : @plan.TensorRef
  tensor : @safetensors.TensorMetadata
} derive(Debug, Eq)

///|
/// Complete ordered weight mapping for one immutable dense Llama model.
pub struct LlamaWeightBindings {
  identity : @spec.ModelIdentity
  bindings : ReadOnlyArray[TensorBinding]
  materialized_bytes : Int64
} derive(Debug, Eq)

///|
pub fn TensorBinding::reference(self : TensorBinding) -> @plan.TensorRef {
  self.reference
}

///|
pub fn TensorBinding::tensor(
  self : TensorBinding,
) -> @safetensors.TensorMetadata {
  self.tensor
}

///|
pub fn LlamaWeightBindings::identity(
  self : LlamaWeightBindings,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn LlamaWeightBindings::length(self : LlamaWeightBindings) -> Int {
  self.bindings.length()
}

///|
pub fn LlamaWeightBindings::binding(
  self : LlamaWeightBindings,
  reference : @plan.TensorRef,
) -> TensorBinding? {
  self.bindings.get(reference.as_int())
}

///|
pub fn LlamaWeightBindings::materialized_bytes(
  self : LlamaWeightBindings,
) -> Int64 {
  self.materialized_bytes
}
