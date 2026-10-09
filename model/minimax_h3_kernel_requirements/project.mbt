///|
/// MiniMax adapter failures preserve only typed startup errors.
pub(all) suberror MiniMaxH3KernelRequirementsError {
  PlanFailure(@joint_diffusion_plan.JointDiffusionPlanError)
  RequirementsFailure(
    @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsError
  )
} derive(Debug, Eq)

///|
/// Projects verified MiniMax H3 metadata through its exact joint diffusion plan
/// into family-neutral AOT requirements. No artifact availability is claimed.
pub fn project(
  metadata : @minimax_h3_spec.MiniMaxH3ModelMetadata,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements raise MiniMaxH3KernelRequirementsError {
  let plan = @minimax_h3.build(metadata) catch {
    error => raise PlanFailure(error)
  }
  let limits = try! @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementLimits::new(
    maximum_transformer_layers=64,
    maximum_requirements=33,
  )
  @joint_diffusion_kernel_requirements.derive_requirements(plan, limits~) catch {
    error => raise RequirementsFailure(error)
  }
}
