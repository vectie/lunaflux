///|
pub(all) suberror ExecutionError {
  InvalidFrameCount
  Closing
  ReentrantClose
}

///|
/// A startup-bound frame: joint prediction from the old states, then both RF
/// updates, with immutable operands and step-specific coefficients. execute
/// must await completion. drain must make release safe even after partial
/// enqueue failure. release must be retryable on failure; success is terminal.
/// Ownership transfers to Execution only when its constructor succeeds.
pub(open) trait PreparedFrame {
  fn execute(Self) -> Unit raise
  fn drain(Self) -> Unit raise
  fn release(Self) -> Unit raise
}

///|
priv struct Frames[F] {
  values : ReadOnlyArray[F]
  mut executing : Bool
}

///|
impl[F : PreparedFrame] @joint_diffusion_loop.JointBackend for Frames[F] with fn execute_step(
  self,
  index,
  _step,
) {
  self.executing = true
  defer {
    self.executing = false
  }
  self.values[index].execute()
}

///|
/// Single-owner request execution. Frames are prebound before the first step;
/// no launch lists, coefficients, hashes or tensor storage are created here.
/// The caller must always close this owner, including after cancellation/error.
pub struct Execution[F] {
  priv driver : @joint_diffusion_loop.Loop
  priv frames : Frames[F]
  priv mut closing : Bool
  priv mut releasing : Bool
  priv mut released : Int
  priv mut drained : Bool
}

///|
/// Transfer exclusive frame ownership on success. The array is copied once;
/// frame handles must not appear twice or be used/released by another owner.
/// Binding frame i to plan.steps()[i] is the preparation layer's obligation.
pub fn[F] Execution::new(
  plan : @joint_diffusion_loop.PreparedPlan,
  frames : ArrayView[F],
) -> Execution[F] raise ExecutionError {
  guard frames.length() == plan.steps().length() else {
    raise InvalidFrameCount
  }
  {
    driver: plan.start(),
    frames: {
      values: ReadOnlyArray::from_array(frames.to_owned()),
      executing: false,
    },
    closing: false,
    releasing: false,
    released: 0,
    drained: false,
  }
}

///|
pub fn[F : PreparedFrame] Execution::advance(
  self : Execution[F],
) -> @joint_diffusion_loop.Progress raise {
  guard !self.closing else { raise Closing }
  self.driver.advance(self.frames)
}

///|
pub fn[F] Execution::cancel(self : Execution[F]) -> Unit {
  self.driver.cancel()
}

///|
pub fn[F] Execution::progress(
  self : Execution[F],
) -> @joint_diffusion_loop.Progress {
  self.driver.progress()
}

///|
pub fn[F] Execution::completed_steps(self : Execution[F]) -> Int {
  self.driver.completed_steps()
}

///|
pub fn[F] Execution::is_closed(self : Execution[F]) -> Bool {
  self.released == self.frames.values.length()
}

///|
/// Close at a synchronous boundary. Stop submissions first, then drain and
/// release each frame. On failure retain the exact cursor and all remaining
/// handles: calling close again retries without releasing successful frames
/// twice. No GC-only device release or swallowed cleanup failure.
pub fn[F : PreparedFrame] Execution::close(self : Execution[F]) -> Unit raise {
  guard !self.frames.executing && !self.releasing else { raise ReentrantClose }
  self.closing = true
  self.driver.cancel()
  self.releasing = true
  defer {
    self.releasing = false
  }
  while self.released < self.frames.values.length() {
    let frame = self.frames.values[self.released]
    if !self.drained {
      frame.drain()
      self.drained = true
    }
    frame.release()
    self.released += 1
    self.drained = false
  }
}
