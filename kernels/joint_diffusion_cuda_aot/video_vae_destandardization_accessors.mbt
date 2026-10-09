///|
pub fn JointDiffusionVideoVaeLatentDestandardizationCudaCandidate::numerical_contract(
  self : Self,
) -> JointDiffusionCudaNumericalContract {
  self.numerical_contract
}

///|
pub fn JointDiffusionVideoVaeLatentDestandardizationCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionVideoVaeLatentDestandardizationCudaCandidate::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionVideoVaeLatentDestandardizationCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointDiffusionVideoVaeLatentDestandardizationCudaCandidate::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionVideoVaeLatentDestandardizationCudaCandidate::capability(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAotCapability {
  self.capability
}

///|
pub fn JointDiffusionVideoVaeLatentDestandardizationCudaCandidate::parameter_digest(
  self : Self,
) -> @joint_diffusion_vae_reference.JointVaeLatentDestandardizationDigest {
  self.parameter_digest
}

///|
pub fn JointDiffusionVideoVaeLatentDestandardizationCudaCandidate::request_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request_shape
}

///|
pub fn JointDiffusionVideoVaeLatentDestandardizationCudaCandidate::channel_count(
  self : Self,
) -> Int {
  self.channel_count
}

///|
pub fn JointDiffusionVideoVaeLatentDestandardizationCudaCandidate::batch_count(
  self : Self,
) -> Int {
  self.batch_count
}

///|
pub fn JointDiffusionVideoVaeLatentDestandardizationCudaCandidate::elements_per_channel(
  self : Self,
) -> Int64 {
  self.elements_per_channel
}

///|
pub fn JointDiffusionVideoVaeLatentDestandardizationCudaCandidate::element_count(
  self : Self,
) -> Int64 {
  self.element_count
}

///|
pub fn JointDiffusionVideoVaeLatentDestandardizationCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn JointDiffusionVideoVaeLatentDestandardizationCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn JointDiffusionVideoVaeLatentDestandardizationCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionVideoVaeLatentDestandardizationCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn JointDiffusionVideoVaeLatentDestandardizationCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn JointDiffusionVideoVaeLatentDestandardizationCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn JointDiffusionVideoVaeLatentDestandardizationCudaCandidate::source_digest(
  self : Self,
) -> JointDiffusionCudaSourceDigest {
  self.source_digest
}

///|
pub fn JointDiffusionVideoVaeLatentDestandardizationCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn JointDiffusionVideoVaeLatentDestandardizationCudaCandidate::recipe_digest(
  self : Self,
) -> JointDiffusionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn JointDiffusionVideoVaeLatentDestandardizationCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}
