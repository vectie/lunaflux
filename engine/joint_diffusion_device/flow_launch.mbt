///|
/// Startup binding for the admitted F32 eta-zero update artifact. The caller
/// must supply the function compiled for this latent extent and block size.
/// Both modality predictions must finish before either update is submitted.
/// The coefficient table must already have been uploaded. No device resource
/// is allocated here; the ordered queue subsequently retains the arguments.
pub fn LatentStorage::flow_launch(
  self : Self,
  function : @device.Function,
  step~ : Int,
  latent~ : @joint_diffusion_plan.JointDiffusionLatentKind,
  block_size~ : Int,
) -> @device.OrderedKernelLaunch raise {
  let (state, velocity) = match latent {
    VideoLatent =>
      (
        @joint_diffusion_loop.LatentBuffer::VideoState,
        @joint_diffusion_loop.LatentBuffer::VideoVelocity,
      )
    AudioLatent => (AudioState, AudioVelocity)
  }
  let elements = self.plan.region(state).byte_count / 4L
  let dimensions = linear_dimensions(elements, block_size)
  let offset = self.plan.region(FlowTableBuffer).offset +
    self.plan.flow_table().coefficient_offset(step, latent)
  let coefficients = @device.KernelArgument::device_region(
    self.allocation,
    offset~,
    byte_count=12L,
    alignment=16L,
  )
  let state_argument = self.argument(state)
  let arguments = @device.KernelArguments::new([
    state_argument,
    self.argument(velocity),
    coefficients,
    state_argument,
  ])
  @device.OrderedKernelLaunch::new(function, dimensions, arguments)
}
