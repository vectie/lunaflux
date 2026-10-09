///|
pub fn JointDiffusionOutputProjectionCudaCandidate::numerical_contract(
  self : Self,
) -> JointDiffusionCudaNumericalContract {
  self.numerical_contract
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::capability(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAotCapability {
  self.capability
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::latent(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionLatentKind {
  self.latent
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::request_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request_shape
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::row_count(
  self : Self,
) -> Int64 {
  self.row_count
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::input_width(
  self : Self,
) -> Int {
  self.input_width
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::output_width(
  self : Self,
) -> Int {
  self.output_width
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::input_element_count(
  self : Self,
) -> Int64 {
  self.input_element_count
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::output_element_count(
  self : Self,
) -> Int64 {
  self.output_element_count
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::model_weight_element_count(
  self : Self,
) -> Int64 {
  self.projection_weight_element_count + self.projection_bias_element_count
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::projection_weight_element_count(
  self : Self,
) -> Int64 {
  self.projection_weight_element_count
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::projection_bias_element_count(
  self : Self,
) -> Int64 {
  self.projection_bias_element_count
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::source_digest(
  self : Self,
) -> JointDiffusionCudaSourceDigest {
  self.source_digest
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::recipe_digest(
  self : Self,
) -> JointDiffusionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn JointDiffusionOutputProjectionCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}
