///|
/// Compose already prepared, heterogeneous conditioning and decoder owners
/// with the denoising queues. No module needs to expose or reconstruct its
/// private launch list. Ownership of supplied stages transfers after initial
/// validation; retain this owner after any later error and close it.
pub fn PipelineProgram::prepare_stages(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  plan : @joint_diffusion_loop.PreparedPlan,
  storage : LatentStorage,
  conditioning : ArrayView[@joint_diffusion_execution.PreparedStage],
  predictions : ArrayView[ReadOnlyArray[@device.OrderedKernelLaunch]],
  video_decode : @joint_diffusion_execution.PreparedStage,
  audio_decode : @joint_diffusion_execution.PreparedStage,
  video_update~ : @device.Function,
  audio_update~ : @device.Function,
  video_block_size~ : Int,
  audio_block_size~ : Int,
) -> Unit raise {
  guard !self.started &&
    !self.closed &&
    !conditioning.is_empty() &&
    predictions.length() == plan.steps().length() &&
    storage.matches_prepared(plan) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  for launches in predictions {
    guard !launches.is_empty() else {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
    }
  }
  self.started = true
  self.pending.append(conditioning)
  // Retain both consuming owners before a fallible queue creation. Insert new
  // prediction queues before the decoders so failure cleanup remains reverse
  // dependency order: audio, video, denoising, conditioning.
  self.pending.push(video_decode)
  self.pending.push(audio_decode)
  for step, launches in predictions {
    let frame = KernelFrame::denoise(
      context,
      stream,
      storage,
      launches[:],
      step~,
      video_update~,
      audio_update~,
      video_block_size~,
      audio_block_size~,
    )
    self.pending.insert(
      conditioning.length() + step,
      @joint_diffusion_execution.PreparedStage::own(frame),
    )
  }
  let c = conditioning.length()
  let d = predictions.length()
  self.pipeline = Some(
    @joint_diffusion_execution.Pipeline::new(
      plan,
      self.pending[0:c],
      self.pending[c:c + d],
      self.pending[c + d],
      self.pending[c + d + 1],
    ),
  )
  self.pending.clear()
}
