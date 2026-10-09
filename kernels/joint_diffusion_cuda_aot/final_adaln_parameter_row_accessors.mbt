///|
pub fn JointDiffusionFinalAdaLnParameterRowCudaCandidate::numerical_contract(
  self : Self,
) -> JointDiffusionCudaNumericalContract {
  self.numerical_contract
}

///|
pub fn JointDiffusionFinalAdaLnParameterRowCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionFinalAdaLnParameterRowCudaCandidate::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionFinalAdaLnParameterRowCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointDiffusionFinalAdaLnParameterRowCudaCandidate::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionFinalAdaLnParameterRowCudaCandidate::capability(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAotCapability {
  self.capability
}

///|
pub fn JointDiffusionFinalAdaLnParameterRowCudaCandidate::input_width(
  self : Self,
) -> Int {
  self.input_width
}

///|
pub fn JointDiffusionFinalAdaLnParameterRowCudaCandidate::hidden_size(
  self : Self,
) -> Int {
  self.hidden_size
}

///|
pub fn JointDiffusionFinalAdaLnParameterRowCudaCandidate::weight_element_count(
  self : Self,
) -> Int64 {
  self.weight_element_count
}

///|
pub fn JointDiffusionFinalAdaLnParameterRowCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn JointDiffusionFinalAdaLnParameterRowCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn JointDiffusionFinalAdaLnParameterRowCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionFinalAdaLnParameterRowCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn JointDiffusionFinalAdaLnParameterRowCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn JointDiffusionFinalAdaLnParameterRowCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn JointDiffusionFinalAdaLnParameterRowCudaCandidate::source_digest(
  self : Self,
) -> JointDiffusionCudaSourceDigest {
  self.source_digest
}

///|
pub fn JointDiffusionFinalAdaLnParameterRowCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn JointDiffusionFinalAdaLnParameterRowCudaCandidate::recipe_digest(
  self : Self,
) -> JointDiffusionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn JointDiffusionFinalAdaLnParameterRowCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}
