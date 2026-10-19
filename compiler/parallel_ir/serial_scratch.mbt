///|
/// Storage for explicitly serialized execution phases. Each phase must retire
/// every reader before the next phase begins; concurrent/forked phases require
/// different arenas. This is transient storage, never persistent model state.
pub struct SerialScratch {
  priv peak : Int64
  priv saved : Int64
}

///|
pub fn SerialScratch::new(sizes : ArrayView[Int64]) -> Self raise ParallelError {
  for size in sizes; peak = 0L, total = 0L {
    guard size >= 0L && size <= 9223372036854775807L - total else {
      raise Capacity
    }
    continue Int64::max(peak, size), total + size
  } nobreak {
    { peak, saved: total - peak, }
  }
}

///|
pub fn SerialScratch::bytes(self : Self) -> Int64 {
  self.peak
}

///|
pub fn SerialScratch::saved_bytes(self : Self) -> Int64 {
  self.saved
}
