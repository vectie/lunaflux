///|
/// Payload-safe failures from the operating system monotonic clock.
pub(all) suberror MonotonicClockError {
  Unavailable
  OutOfRange
} derive(Debug, Eq)

///|
/// Opaque access to the operating system monotonic clock. This value owns no
/// native resource and requires no close operation.
#valtype
pub struct MonotonicClock {
  priv marker : Unit
}

///|
pub struct PreparedMonotonicRead {
  priv clock : MonotonicClock
  priv output : Ref[UInt64]
  priv mut used : Bool
  priv mut ready : Bool
}

///|
#valtype
pub(all) enum MonotonicReadStatus {
  MonotonicReadReady
  MonotonicReadFailed
} derive(Debug, Eq)

///|
/// Constructs the process-local operating system monotonic clock capability.
pub fn MonotonicClock::system() -> MonotonicClock {
  { marker: () }
}

///|
pub fn MonotonicClock::prepare_read(
  self : MonotonicClock,
) -> PreparedMonotonicRead {
  { clock: self, output: Ref(0UL), used: false, ready: false }
}

///|
pub fn PreparedMonotonicRead::read_status(
  self : PreparedMonotonicRead,
) -> MonotonicReadStatus {
  ignore(self.clock.marker)
  if self.used {
    return MonotonicReadFailed
  }
  self.used = true
  if @native.now_millis_status(self.output) == 0 {
    self.ready = true
    MonotonicReadReady
  } else {
    MonotonicReadFailed
  }
}

///|
pub fn PreparedMonotonicRead::value(
  self : PreparedMonotonicRead,
) -> UInt64 raise MonotonicClockError {
  if !self.ready {
    raise Unavailable
  }
  self.output.val
}

///|
fn map_native_error(
  error : @native.MonotonicClockNativeError,
) -> MonotonicClockError {
  match error {
    NativeUnavailable => Unavailable
    NativeOutOfRange => OutOfRange
  }
}

///|
/// Returns elapsed monotonic milliseconds from an unspecified origin. The
/// value is suitable for intervals and deadlines, never civil or wall time.
pub fn MonotonicClock::now_millis(
  self : MonotonicClock,
) -> UInt64 raise MonotonicClockError {
  ignore(self.marker)
  @native.now_millis() catch {
    error => raise map_native_error(error)
  }
}
