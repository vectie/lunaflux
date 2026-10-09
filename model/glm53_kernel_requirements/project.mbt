///|
pub(all) suberror Glm53KernelRequirementsError {
  RequirementsFailure(
    @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsError
  )
} derive(Debug, Eq)

///|
fn official_limits() -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementLimits {
  try! @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementLimits::new(
    maximum_layers=128,
    maximum_vision_layers=64,
    maximum_requirements=128,
  )
}

///|
pub fn build_full_kernel_requirements(
  metadata : @glm53_spec.Glm53FullMetadata,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements raise Glm53KernelRequirementsError {
  @glm_hybrid_kernel_requirements.derive_requirements(
    @glm53.build_full_execution_plan(metadata),
    limits=official_limits(),
  ) catch {
    error => raise RequirementsFailure(error)
  }
}

///|
pub fn build_flash_kernel_requirements(
  metadata : @glm53_spec.Glm53FlashMetadata,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements raise Glm53KernelRequirementsError {
  @glm_hybrid_kernel_requirements.derive_requirements(
    @glm53.build_flash_execution_plan(metadata),
    limits=official_limits(),
  ) catch {
    error => raise RequirementsFailure(error)
  }
}
