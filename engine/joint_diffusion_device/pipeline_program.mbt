///|
/// Prepares a complete conditioning -> paired denoising -> video/audio decode
/// request with one stream. Owns executor leases, not model/latent/output buffers.
/// Retain after any prepare failure and call close before closing buffer owners.
pub struct PipelineProgram {
  priv pending : Array[@joint_diffusion_execution.PreparedStage]
  priv mut pending_drained : Bool
  priv mut pipeline : @joint_diffusion_execution.Pipeline[
    @joint_diffusion_execution.PreparedStage,
  ]?
  priv mut started : Bool
  priv mut closed : Bool
}

///|
pub fn PipelineProgram::new() -> Self {
  {
    pending: [],
    pending_drained: false,
    pipeline: None,
    started: false,
    closed: false,
  }
}

///|
/// All launch bindings are prepared by their module owners. Each prediction
/// includes timestep, input packing, full transformer and output heads; this
/// method appends both RF updates and never treats missing modules as no-ops.
pub fn PipelineProgram::prepare(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  plan : @joint_diffusion_loop.PreparedPlan,
  storage : LatentStorage,
  conditioning : ArrayView[ReadOnlyArray[@device.OrderedKernelLaunch]],
  predictions : ArrayView[ReadOnlyArray[@device.OrderedKernelLaunch]],
  video_decode : ArrayView[@device.OrderedKernelLaunch],
  audio_decode : ArrayView[@device.OrderedKernelLaunch],
  video_update~ : @device.Function,
  audio_update~ : @device.Function,
  video_block_size~ : Int,
  audio_block_size~ : Int,
) -> Unit raise {
  guard !self.started &&
    !self.closed &&
    !conditioning.is_empty() &&
    predictions.length() == plan.steps().length() &&
    !video_decode.is_empty() &&
    !audio_decode.is_empty() else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  guard storage.matches_prepared(plan) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  for launches in conditioning {
    guard !launches.is_empty() else {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
    }
  }
  for launches in predictions {
    guard !launches.is_empty() else {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
    }
  }
  self.started = true
  for launches in conditioning {
    self.pending.push(
      @joint_diffusion_execution.PreparedStage::own(
        KernelFrame::new(context, stream, launches[:]),
      ),
    )
  }
  for step, launches in predictions {
    self.pending.push(
      @joint_diffusion_execution.PreparedStage::own(
        KernelFrame::denoise(
          context,
          stream,
          storage,
          launches[:],
          step~,
          video_update~,
          audio_update~,
          video_block_size~,
          audio_block_size~,
        ),
      ),
    )
  }
  self.pending.push(
    @joint_diffusion_execution.PreparedStage::own(
      KernelFrame::new(context, stream, video_decode),
    ),
  )
  self.pending.push(
    @joint_diffusion_execution.PreparedStage::own(
      KernelFrame::new(context, stream, audio_decode),
    ),
  )
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

///|
pub fn PipelineProgram::advance(
  self : Self,
) -> @joint_diffusion_execution.PipelineStage raise {
  guard !self.closed && self.pipeline is Some(pipeline) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  pipeline.advance()
}

///|
pub fn PipelineProgram::cancel(self : Self) -> Unit {
  if self.pipeline is Some(pipeline) {
    pipeline.cancel()
  }
}

///|
pub fn PipelineProgram::outputs_ready(self : Self) -> Bool {
  !self.closed &&
  self.pipeline is Some(pipeline) &&
  pipeline.stage() == Completed
}

///|
pub fn PipelineProgram::close(self : Self) -> Unit raise {
  self.closed = true
  if self.pipeline is Some(pipeline) {
    pipeline.close()
    self.pipeline = None
  }
  while !self.pending.is_empty() {
    let frame = self.pending[self.pending.length() - 1]
    if !self.pending_drained {
      @joint_diffusion_execution.PreparedFrame::drain(frame)
      self.pending_drained = true
    }
    @joint_diffusion_execution.PreparedFrame::release(frame)
    self.pending_drained = false
    let _ = self.pending.pop()
  }
}
