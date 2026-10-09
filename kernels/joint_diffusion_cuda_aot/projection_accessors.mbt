///|
pub fn JointDiffusionLatentProjectionCudaCandidate::numerical_contract(
  self : Self,
) -> JointDiffusionCudaNumericalContract {
  self.numerical_contract
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::capability(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAotCapability {
  self.capability
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::latent(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionLatentKind {
  self.latent
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::request_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request_shape
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::row_count(
  self : Self,
) -> Int64 {
  self.row_count
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::input_width(
  self : Self,
) -> Int {
  self.input_width
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::output_width(
  self : Self,
) -> Int {
  self.output_width
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::input_element_count(
  self : Self,
) -> Int64 {
  self.input_element_count
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::output_element_count(
  self : Self,
) -> Int64 {
  self.output_element_count
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::model_weight_element_count(
  self : Self,
) -> Int64 {
  self.projection_weight_element_count + self.projection_bias_element_count
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::projection_weight_element_count(
  self : Self,
) -> Int64 {
  self.projection_weight_element_count
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::projection_bias_element_count(
  self : Self,
) -> Int64 {
  self.projection_bias_element_count
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::source_digest(
  self : Self,
) -> JointDiffusionCudaSourceDigest {
  self.source_digest
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::recipe_digest(
  self : Self,
) -> JointDiffusionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn JointDiffusionLatentProjectionCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}
