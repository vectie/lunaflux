///|
pub fn JointDiffusionCudaAotLimits::new(
  maximum_elements~ : Int64,
  maximum_source_bytes~ : Int,
  maximum_recipe_bytes~ : Int,
) -> Self raise JointDiffusionCudaAotError {
  if maximum_elements <= 0L ||
    maximum_elements > 1073741823L ||
    maximum_source_bytes <= 0 ||
    maximum_recipe_bytes <= 0 {
    raise Invalid(Limit)
  }
  { maximum_elements, maximum_source_bytes, maximum_recipe_bytes, }
}

///|
pub fn JointDiffusionShiftedFlowCudaCandidate::numerical_contract(
  self : Self,
) -> JointDiffusionCudaNumericalContract {
  self.numerical_contract
}

///|
pub fn JointDiffusionShiftedFlowCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionShiftedFlowCudaCandidate::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionShiftedFlowCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointDiffusionShiftedFlowCudaCandidate::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionShiftedFlowCudaCandidate::capability(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAotCapability {
  self.capability
}

///|
pub fn JointDiffusionShiftedFlowCudaCandidate::latent(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionLatentKind {
  self.latent
}

///|
pub fn JointDiffusionShiftedFlowCudaCandidate::request_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request_shape
}

///|
pub fn JointDiffusionShiftedFlowCudaCandidate::element_count(
  self : Self,
) -> Int64 {
  self.element_count
}

///|
pub fn JointDiffusionShiftedFlowCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn JointDiffusionShiftedFlowCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn JointDiffusionShiftedFlowCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionShiftedFlowCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn JointDiffusionShiftedFlowCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn JointDiffusionShiftedFlowCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn JointDiffusionShiftedFlowCudaCandidate::source_digest(
  self : Self,
) -> JointDiffusionCudaSourceDigest {
  self.source_digest
}

///|
pub fn JointDiffusionShiftedFlowCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn JointDiffusionShiftedFlowCudaCandidate::recipe_digest(
  self : Self,
) -> JointDiffusionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn JointDiffusionShiftedFlowCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

///|
pub fn JointDiffusionCudaSourceDigest::as_hex(self : Self) -> String {
  self.0
}

///|
pub fn JointDiffusionCudaRecipeDigest::as_hex(self : Self) -> String {
  self.0
}
