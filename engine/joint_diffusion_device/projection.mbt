///|
/// Request-local projection outputs with a borrowed latent arena. Reuse these
/// outputs across steps after consumers complete; queues hold native leases.
pub struct LatentProjections {
  priv storage : LatentStorage
  priv plan : @joint_diffusion_loop.LatentProjectionPlan
  priv output : @device.Allocation
}

///|
pub fn LatentProjections::allocate(
  context : @device.Context,
  storage : LatentStorage,
  model : @joint_diffusion_plan.JointDiffusionPlan,
  max_bytes~ : Int64,
) -> Self raise {
  let plan = storage.plan.projections(model, max_bytes~)
  { storage, plan, output: context.allocate(plan.byte_count()), }
}

///|
pub fn LatentProjections::output_argument(
  self : Self,
  latent : @joint_diffusion_plan.JointDiffusionLatentKind,
) -> @device.KernelArgument raise @device.DeviceError {
  let region = self.plan.shape(latent).output
  @device.KernelArgument::device_region(
    self.output,
    offset=region.offset,
    byte_count=region.byte_count,
    alignment=256L,
  )
}

///|
/// Startup-bind the existing dense F32-input/F32-weight/F32-bias/BF16-output
/// AOT renderer (256 threads). Function and weights must be admitted for this
/// model/modality. Weight allocations remain caller-owned. No copies occur.
pub fn LatentProjections::launch(
  self : Self,
  function : @device.Function,
  latent~ : @joint_diffusion_plan.JointDiffusionLatentKind,
  weights~ : @device.Allocation,
  weight_offset~ : Int64,
  bias~ : @device.Allocation,
  bias_offset~ : Int64,
) -> @device.OrderedKernelLaunch raise @device.DeviceError {
  let shape = self.plan.shape(latent)
  let input = match latent {
    VideoLatent => @joint_diffusion_loop.LatentBuffer::VideoState
    AudioLatent => AudioState
  }
  let dimensions = linear_dimensions(shape.output.byte_count / 2L, 256)
  let arguments = @device.KernelArguments::new([
    self.storage.argument(input),
    @device.KernelArgument::device_region(
      weights,
      offset=weight_offset,
      byte_count=shape.weight_bytes,
      alignment=16L,
    ),
    @device.KernelArgument::device_region(
      bias,
      offset=bias_offset,
      byte_count=shape.bias_bytes,
      alignment=16L,
    ),
    self.output_argument(latent),
  ])
  @device.OrderedKernelLaunch::new(function, dimensions, arguments)
}

///|
/// Close after all queue leases, before releasing the borrowed latent arena.
/// Failure leaves ownership intact for retry; the latent arena is not closed.
pub fn LatentProjections::close(self : Self) -> Unit raise @device.DeviceError {
  self.output.close()
}

///|
/// Startup-only producer/consumer join: equivalent plans do not imply the
/// same request-local latent values.
pub fn LatentProjections::matches_storage(
  self : Self,
  storage : LatentStorage,
) -> Bool {
  self.storage.allocation.same_storage(storage.allocation)
}
