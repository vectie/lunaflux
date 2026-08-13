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
/// Constructs the process-local operating system monotonic clock capability.
pub fn MonotonicClock::system() -> MonotonicClock {
  { marker: () }
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
