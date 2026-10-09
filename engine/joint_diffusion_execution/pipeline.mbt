///|
pub(all) enum PipelineStage {
  Conditioning
  Denoising
  VideoDecode
  AudioDecode
  Completed
  Cancelled
  Failed
  Closed
} derive(Eq, Debug)

///|
pub(all) suberror PipelineError {
  MissingConditioning
  InvalidDenoisingCount
  Terminal
  Reentrant
}

///|
/// Single-owner whole-request ordering. Each frame must finish its device work
/// before returning; output is usable only after both decoders complete.
/// No allocations or launch-list construction occur in advance.
pub struct Pipeline[F] {
  priv frames : ReadOnlyArray[F]
  priv conditioning_count : Int
  priv denoising_count : Int
  priv mut completed : Int
  priv mut stage : PipelineStage
  priv mut busy : Bool
  priv mut release_remaining : Int
  priv mut current_drained : Bool
  priv mut closing : Bool
}

///|
/// Transfer frame ownership only on success. Distinct frame handles and their
/// buffers are prebound; denoising i must include both updates for plan step i.
pub fn[F] Pipeline::new(
  plan : @joint_diffusion_loop.PreparedPlan,
  conditioning : ArrayView[F],
  denoising : ArrayView[F],
  video_decode : F,
  audio_decode : F,
) -> Pipeline[F] raise PipelineError {
  guard !conditioning.is_empty() else { raise MissingConditioning }
  guard denoising.length() == plan.steps().length() else {
    raise InvalidDenoisingCount
  }
  let frames = conditioning.to_owned()
  frames.append(denoising)
  frames.push(video_decode)
  frames.push(audio_decode)
  {
    frames: ReadOnlyArray::from_array(frames),
    conditioning_count: conditioning.length(),
    denoising_count: denoising.length(),
    completed: 0,
    stage: Conditioning,
    busy: false,
    release_remaining: frames.length(),
    current_drained: false,
    closing: false,
  }
}

///|
pub fn[F] Pipeline::stage(self : Pipeline[F]) -> PipelineStage {
  self.stage
}

///|
pub fn[F] Pipeline::completed_frames(self : Pipeline[F]) -> Int {
  self.completed
}

///|
pub fn[F : PreparedFrame] Pipeline::advance(
  self : Pipeline[F],
) -> PipelineStage raise {
  guard !self.busy else { raise PipelineError::Reentrant }
  guard !self.closing &&
    self.stage is (Conditioning | Denoising | VideoDecode | AudioDecode) else {
    raise PipelineError::Terminal
  }
  self.busy = true
  defer {
    self.busy = false
  }
  self.frames[self.completed].execute() catch {
    error => {
      self.stage = Failed
      raise error
    }
  }
  self.completed += 1
  if self.stage == Cancelled {
    return Cancelled
  }
  self.stage = if self.completed < self.conditioning_count {
    Conditioning
  } else if self.completed < self.conditioning_count + self.denoising_count {
    Denoising
  } else if self.completed == self.frames.length() - 2 {
    VideoDecode
  } else if self.completed == self.frames.length() - 1 {
    AudioDecode
  } else {
    Completed
  }
  self.stage
}

///|
/// Cooperative cancellation stops the next frame; close drains and releases.
pub fn[F] Pipeline::cancel(self : Pipeline[F]) -> Unit {
  if self.stage is (Conditioning | Denoising | VideoDecode | AudioDecode) {
    self.stage = Cancelled
  }
}

///|
/// Release consuming frames in reverse order, retaining the exact retry cursor.
/// External allocation/module owners may close only after this succeeds.
pub fn[F : PreparedFrame] Pipeline::close(self : Pipeline[F]) -> Unit raise {
  guard !self.busy else { raise PipelineError::Reentrant }
  if self.stage == Closed {
    return
  }
  self.busy = true
  defer {
    self.busy = false
  }
  self.closing = true
  self.cancel()
  while self.release_remaining > 0 {
    let frame = self.frames[self.release_remaining - 1]
    if !self.current_drained {
      frame.drain()
      self.current_drained = true
    }
    frame.release()
    self.current_drained = false
    self.release_remaining -= 1
  }
  self.stage = Closed
}
