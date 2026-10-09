///|
/// Join compiled output extents to the actual request before preparing queues.
pub fn LatentStorage::validate_prediction_rows(
  self : Self,
  model : @joint_diffusion_plan.JointDiffusionPlan,
  video_rows~ : Int,
  audio_rows~ : Int,
) -> Unit raise {
  let plan = self.plan.projections(model, max_bytes=9223372036854775807L)
  guard plan.shape(VideoLatent).rows == video_rows.to_int64() &&
    plan.shape(AudioLatent).rows == audio_rows.to_int64() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
}

///|
/// F32 selected hidden -> F32 latent velocity, using the existing dense F32
/// weight/bias/output renderer. Writes directly into the flow-update input;
/// no latent copy or conversion is inserted. Both heads must complete before
/// either modality's flow update. This binds a prediction, not CFG guidance.
pub fn LatentStorage::output_projection_launch(
  self : Self,
  function : @device.Function,
  model : @joint_diffusion_plan.JointDiffusionPlan,
  latent~ : @joint_diffusion_plan.JointDiffusionLatentKind,
  hidden~ : @device.Allocation,
  hidden_offset~ : Int64,
  weights~ : @device.Allocation,
  weight_offset~ : Int64,
  bias~ : @device.Allocation,
  bias_offset~ : Int64,
) -> @device.OrderedKernelLaunch raise {
  // The forward projection plan checks this request's model identity. Reversing
  // its dimensions yields the output head, while preserving the F32 boundary.
  let shape = self.plan
    .projections(model, max_bytes=9223372036854775807L)
    .shape(latent)
  let (input_bytes, output_bytes, bias_bytes) = output_projection_bytes(
    shape.rows,
    shape.output_width,
    shape.input_width,
  )
  let velocity = match latent {
    VideoLatent => @joint_diffusion_loop.LatentBuffer::VideoVelocity
    AudioLatent => AudioVelocity
  }
  guard self.plan.region(velocity).byte_count == output_bytes else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let arguments = @device.KernelArguments::new([
    @device.KernelArgument::device_region(
      hidden,
      offset=hidden_offset,
      byte_count=input_bytes,
      alignment=4L,
    ),
    @device.KernelArgument::device_region(
      weights,
      offset=weight_offset,
      byte_count=shape.weight_bytes,
      alignment=4L,
    ),
    @device.KernelArgument::device_region(
      bias,
      offset=bias_offset,
      byte_count=bias_bytes,
      alignment=4L,
    ),
    self.argument(velocity),
  ])
  @device.OrderedKernelLaunch::new(
    function,
    linear_dimensions(output_bytes / 4L, 256),
    arguments,
  )
}

///|
fn output_projection_bytes(
  rows : Int64,
  hidden : Int,
  latent : Int,
) -> (Int64, Int64, Int64) raise @device.DeviceError {
  guard rows > 0L &&
    hidden > 0 &&
    latent > 0 &&
    rows <= 2147483647L / hidden.to_int64() &&
    rows <= 2147483647L / latent.to_int64() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  (
    rows * hidden.to_int64() * 4L,
    rows * latent.to_int64() * 4L,
    latent.to_int64() * 4L,
  )
}
