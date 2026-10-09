///|
/// Reusable local clock scratch for cooperative watchdog loops. Unlike a
/// PreparedMonotonicRead, this owner can be polled repeatedly without allocating
/// another scalar cell. A failed poll invalidates the previous observation.
pub struct ReusableMonotonicRead {
  priv output : Ref[UInt64]
  priv mut ready : Bool
}

///|
pub fn MonotonicClock::prepare_poller(
  self : MonotonicClock,
) -> ReusableMonotonicRead {
  ignore(self.marker)
  { output: Ref(0UL), ready: false, }
}

///|
pub fn ReusableMonotonicRead::poll(self : ReusableMonotonicRead) -> Bool {
  self.ready = @native.now_millis_status(self.output) == 0
  self.ready
}

///|
pub fn ReusableMonotonicRead::value(
  self : ReusableMonotonicRead,
) -> UInt64 raise MonotonicClockError {
  if !self.ready {
    raise Unavailable
  }
  self.output.val
}
