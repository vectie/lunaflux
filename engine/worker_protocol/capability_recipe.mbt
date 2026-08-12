///|
/// Immutable startup recipe for one worker row's exact ordered semantic
/// capability list.
///
/// Construction defensively copies the model plan's capabilities. Reusing a
/// recipe never exposes the model-plan package to scheduler policy. Duplicate
/// IDs are preserved deliberately: a plan may require the same semantic
/// operation at multiple ordered positions, and this protocol must not infer
/// that either occurrence is redundant.
///
/// This type deliberately has no `Debug` implementation. Its contents are
/// copied only into authenticated plan-row drafts.
pub struct RowCapabilityRecipe {
  priv identity : @spec.ModelIdentity
  priv model_plan_generation : ModelPlanGeneration
  priv capabilities : ReadOnlyArray[@plan.KernelCapabilityId]
} derive(Eq)

///|
/// Validates and owns a bounded, nonempty recipe. This is a startup operation
/// and allocates the recipe's immutable defensive copy once.
pub fn RowCapabilityRecipe::new(
  identity~ : @spec.ModelIdentity,
  model_plan_generation~ : ModelPlanGeneration,
  capabilities : ArrayView[@plan.KernelCapabilityId],
  limits : WorkerProtocolLimits,
) -> RowCapabilityRecipe raise WorkerProtocolError {
  if capabilities.length() == 0 {
    raise InvalidCapabilityRecipe(EmptyCapabilityRecipe)
  }
  if capabilities.length() > limits.max_capabilities() {
    raise BoundExceeded(Capabilities)
  }
  {
    identity,
    model_plan_generation,
    capabilities: ReadOnlyArray::from_array(capabilities),
  }
}

///|
/// Returns the exact immutable model identity that produced this recipe.
pub fn RowCapabilityRecipe::identity(
  self : RowCapabilityRecipe,
) -> @spec.ModelIdentity {
  self.identity
}

///|
/// Returns the loaded-model generation for which this recipe is valid.
pub fn RowCapabilityRecipe::model_plan_generation(
  self : RowCapabilityRecipe,
) -> ModelPlanGeneration {
  self.model_plan_generation
}

///|
/// Returns the exact number of capability cells copied for each application.
pub fn RowCapabilityRecipe::length(self : RowCapabilityRecipe) -> Int {
  self.capabilities.length()
}

///|
/// Computes the exact flattened capability-table cells required by `rows`
/// applications. It performs checked bounded arithmetic without allocating.
pub fn RowCapabilityRecipe::required_cells(
  self : RowCapabilityRecipe,
  rows : Int,
  limits : WorkerProtocolLimits,
) -> Int raise WorkerProtocolError {
  if rows < 0 || rows > limits.max_plan_rows() {
    raise InvalidLimit(PlanRows)
  }
  let length = self.capabilities.length()
  if rows > limits.max_capabilities() / length {
    raise BoundExceeded(Capabilities)
  }
  rows * length
}

///|
/// Appends the recipe's exact ordered capability list to an authenticated row
/// draft. Capacity is proved before the first write, so failure cannot append
/// a partial recipe. Successful applications copy directly into startup-
/// allocated plan storage and do not allocate.
pub fn PlanRowDraft::push_recipe(
  self : PlanRowDraft,
  recipe : RowCapabilityRecipe,
) -> Unit raise WorkerProtocolError {
  let owner = self.owner()
  let length = recipe.capabilities.length()
  if length > owner.limits.max_capabilities() - owner.capability_count {
    raise BoundExceeded(Capabilities)
  }
  for index in 0..<length {
    owner.capabilities[owner.capability_count + index] = recipe.capabilities[index]
  }
  owner.capability_count = owner.capability_count + length
}
