///|
/// One joint evaluation followed by separate video/audio Euler updates.
pub(all) struct JointStep {
  video_timestep : Double
  audio_timestep : Double
  video_sigma : Double
  audio_sigma : Double
  video_next_sigma : Double
  audio_next_sigma : Double
} derive(Debug, Eq)

///|
pub extend JointStep with @debug.Debug::{to_repr}

///|
pub extend JointStep with Eq::{equal, not_equal}

///|
pub(all) enum Progress {
  Running
  Completed
  Cancelled
  Failed
} derive(Debug, Eq)

///|
pub extend Progress with @debug.Debug::{to_repr}

///|
pub extend Progress with Eq::{equal, not_equal}

///|
pub(all) suberror LoopError {
  InvalidSchedule
  NotRunning
  ReentrantProgress
} derive(Debug, Eq)

///|
pub extend LoopError with @debug.Debug::{to_repr}

///|
pub extend LoopError with Eq::{equal, not_equal}

///|
/// Borrowed prepared storage. Evaluate both predictions from the old joint
/// state, apply both updates, and await completion before returning. No tensor
/// allocation, compilation or conditioning recomputation belongs here.
/// A raised error may leave storage partially updated: the driver never retries.
pub(open) trait JointBackend {
  fn execute_step(Self, Int, JointStep) -> Unit raise
}

///|
/// Single-owner synchronous driver. Device resources remain backend-owned.
pub struct Loop {
  priv steps : ReadOnlyArray[JointStep]
  priv mut completed : Int
  priv mut progress : Progress
  priv mut executing : Bool
}

///|
pub fn Loop::new(steps : ArrayView[JointStep]) -> Loop raise LoopError {
  guard steps.length() > 0 else { raise InvalidSchedule }
  for index, step in steps {
    for
      value in [
        step.video_timestep,
        step.audio_timestep,
        step.video_sigma,
        step.audio_sigma,
        step.video_next_sigma,
        step.audio_next_sigma,
      ] {
      guard !value.is_nan() && !value.is_inf() else { raise InvalidSchedule }
    }
    guard step.video_sigma > 0.0 &&
      step.video_sigma <= 1.0 &&
      step.audio_sigma > 0.0 &&
      step.audio_sigma <= 1.0 &&
      Float::from_double(step.video_sigma) > 0.0 &&
      Float::from_double(step.audio_sigma) > 0.0 &&
      step.video_next_sigma >= 0.0 &&
      step.video_next_sigma < step.video_sigma &&
      step.audio_next_sigma >= 0.0 &&
      step.audio_next_sigma < step.audio_sigma &&
      step.video_timestep == 1.0 - step.video_sigma &&
      step.audio_timestep == 1.0 - step.audio_sigma else {
      raise InvalidSchedule
    }
    if index > 0 {
      guard step.video_sigma == steps[index - 1].video_next_sigma &&
        step.audio_sigma == steps[index - 1].audio_next_sigma else {
        raise InvalidSchedule
      }
    }
  }
  let last = steps[steps.length() - 1]
  guard last.video_next_sigma == 0.0 && last.audio_next_sigma == 0.0 else {
    raise InvalidSchedule
  }
  {
    steps: ReadOnlyArray::from_array(steps.to_owned()),
    completed: 0,
    progress: Running,
    executing: false,
  }
}

///|
/// Execute at most one complete joint step. Backend failures propagate, with
/// progress permanently failed even if only one modality was written.
pub fn[B : JointBackend] Loop::advance(
  self : Loop,
  backend : B,
) -> Progress raise {
  guard !self.executing else { raise ReentrantProgress }
  guard self.progress == Running else { raise NotRunning }
  self.executing = true
  errdefer {
    self.executing = false
    self.progress = Failed
  }
  backend.execute_step(self.completed, self.steps[self.completed])
  self.executing = false
  self.completed += 1
  if self.progress == Running && self.completed == self.steps.length() {
    self.progress = Completed
  }
  self.progress
}

///|
/// Cancellation is sticky. During a synchronous callback it prevents the next
/// submission, but does not claim to interrupt or release the current work.
pub fn Loop::cancel(self : Loop) -> Unit {
  if self.progress == Running {
    self.progress = Cancelled
  }
}

///|
pub fn Loop::progress(self : Loop) -> Progress {
  self.progress
}

///|
pub fn Loop::completed_steps(self : Loop) -> Int {
  self.completed
}
