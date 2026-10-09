///|
/// Family adapter errors preserve only typed startup failures.
pub(all) suberror DeepSeekV4KernelRequirementsError {
  IdentityMismatch
  RequirementsFailure(
    @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsError
  )
} derive(Debug, Eq)

///|
/// Projects one exact DeepSeek-V4 semantic plan into family-neutral capability
/// requirements. This does not select or claim the existence of AOT artifacts.
pub fn project(
  semantic : @deepseek_v4.DeepSeekV4SemanticPlan,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements raise DeepSeekV4KernelRequirementsError {
  let execution = semantic.execution_plan()
  if execution.content_digest() != semantic.content_digest() {
    raise IdentityMismatch
  }
  let limits = try! @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementLimits::new(
    maximum_layers=128,
    maximum_auxiliary_stages=16,
    maximum_requirements=128,
  )
  @advanced_decoder_kernel_requirements.derive_requirements(execution, limits~) catch {
    error => raise RequirementsFailure(error)
  }
}
