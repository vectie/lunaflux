///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::numerical_contract(
  self : Self,
) -> JointDiffusionCudaNumericalContract {
  self.numerical_contract
}

///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::capability(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAotCapability {
  self.capability
}

///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::latent(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionLatentKind {
  self.latent
}

///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::request_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request_shape
}

///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::row_count(self : Self) -> Int64 {
  self.row_count
}

///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::hidden_size(self : Self) -> Int {
  self.hidden_size
}

///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::epsilon_numerator(
  self : Self,
) -> Int {
  self.epsilon_numerator
}

///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::epsilon_denominator(
  self : Self,
) -> Int {
  self.epsilon_denominator
}

///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::epsilon(self : Self) -> Double {
  self.epsilon_numerator.to_double() / self.epsilon_denominator.to_double()
}

///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::element_count(
  self : Self,
) -> Int64 {
  self.element_count
}

///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::weight_element_count(
  self : Self,
) -> Int64 {
  self.weight_element_count
}

///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::source_digest(
  self : Self,
) -> JointDiffusionCudaSourceDigest {
  self.source_digest
}

///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn JointDiffusionFinalRmsNormCudaCandidate::recipe_digest(
  self : Self,
) -> JointDiffusionCudaRecipeDigest {
  self.recipe_digest
}

///|
/// A final-RMSNorm candidate is deliberately not manifest-bindable. It emits
/// BF16 normalized rows only. MiniMax H3 still requires authenticated BF16
/// AdaLN shift/scale modulation and a BF16-to-F32 boundary before either F32
/// output projection can consume those rows.
pub fn JointDiffusionFinalRmsNormCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}
