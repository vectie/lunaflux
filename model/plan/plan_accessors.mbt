///|
pub fn ModelPlan::identity(self : ModelPlan) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn ModelPlan::shape_constraints(self : ModelPlan) -> ShapeConstraints {
  self.shape_constraints
}

///|
pub fn ModelPlan::kv_cache_geometry(self : ModelPlan) -> KvCacheGeometry {
  self.kv_cache_geometry
}

///|
pub fn ModelPlan::kv_execution(self : ModelPlan) -> ModelKvExecution {
  self.kv_execution
}

///|
pub fn ModelPlan::workspace_bounds(self : ModelPlan) -> WorkspaceBounds {
  self.workspace_bounds
}

///|
pub fn ModelPlan::operations(self : ModelPlan) -> ReadOnlyArray[PlanOperation] {
  self.operations
}

///|
pub fn ModelPlan::required_capabilities(
  self : ModelPlan,
) -> ReadOnlyArray[KernelCapabilityId] {
  self.required_capabilities
}

///|
pub fn ModelPlan::final_output(self : ModelPlan) -> ActivationRef {
  self.final_output
}

///|
pub fn ModelPlan::numeric_binding(self : ModelPlan) -> ModelNumericBinding {
  self.numeric_binding
}

///|
pub fn ModelPlan::tensor_count(self : ModelPlan) -> Int {
  self.numeric_binding.tensors().length()
}

///|
pub fn ModelPlan::tensor(
  self : ModelPlan,
  reference : TensorRef,
) -> PlanTensor? {
  guard self.numeric_binding.tensors().get(reference.as_int()) is Some(tensor) &&
    tensor.reference() == reference else {
    return None
  }
  Some(tensor)
}
