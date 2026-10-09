///|
/// Startup-only effect erasure for heterogeneous request stages. The closures
/// are allocated once, not while advancing the request. Ownership of the frame
/// transfers to this handle; do not independently use or release it afterwards.
pub struct PreparedStage {
  priv run : () -> Unit raise
  priv abort : () -> Unit raise
  priv dispose : () -> Unit raise
}

///|
pub fn[F : PreparedFrame] PreparedStage::own(frame : F) -> PreparedStage {
  {
    run: () => frame.execute(),
    abort: () => frame.drain(),
    dispose: () => frame.release(),
  }
}

///|
pub impl PreparedFrame for PreparedStage with fn execute(self) {
  (self.run)()
}

///|
pub impl PreparedFrame for PreparedStage with fn drain(self) {
  (self.abort)()
}

///|
pub impl PreparedFrame for PreparedStage with fn release(self) {
  (self.dispose)()
}
