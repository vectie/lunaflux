///|
pub fn JointDiffusionFinalAdaLnF32CudaCandidate::numerical_contract(
  self : Self,
) -> JointDiffusionCudaNumericalContract {
  self.numerical_contract
}

///|
pub fn JointDiffusionFinalAdaLnF32CudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionFinalAdaLnF32CudaCandidate::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionFinalAdaLnF32CudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointDiffusionFinalAdaLnF32CudaCandidate::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionFinalAdaLnF32CudaCandidate::capability(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAotCapability {
  self.capability
}

///|
pub fn JointDiffusionFinalAdaLnF32CudaCandidate::latent(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionLatentKind {
  self.latent
}

///|
pub fn JointDiffusionFinalAdaLnF32CudaCandidate::request_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request_shape
}

///|
pub fn JointDiffusionFinalAdaLnF32CudaCandidate::row_count(
  self : Self,
) -> Int64 {
  self.row_count
}

///|
pub fn JointDiffusionFinalAdaLnF32CudaCandidate::hidden_size(
  self : Self,
) -> Int {
  self.hidden_size
}

///|
pub fn JointDiffusionFinalAdaLnF32CudaCandidate::element_count(
  self : Self,
) -> Int64 {
  self.element_count
}

///|
pub fn JointDiffusionFinalAdaLnF32CudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn JointDiffusionFinalAdaLnF32CudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn JointDiffusionFinalAdaLnF32CudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionFinalAdaLnF32CudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn JointDiffusionFinalAdaLnF32CudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn JointDiffusionFinalAdaLnF32CudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn JointDiffusionFinalAdaLnF32CudaCandidate::source_digest(
  self : Self,
) -> JointDiffusionCudaSourceDigest {
  self.source_digest
}

///|
pub fn JointDiffusionFinalAdaLnF32CudaCandidate::recipe_digest(
  self : Self,
) -> JointDiffusionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn JointDiffusionFinalAdaLnF32CudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn JointDiffusionFinalAdaLnF32CudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}
