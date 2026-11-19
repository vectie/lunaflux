///|
const NATIVE_OK : Int32 = 0

///|
const NATIVE_UNAVAILABLE : Int32 = 1

///|
const NATIVE_OUT_OF_RANGE : Int32 = 2

///|
/// Payload-free internal failure vocabulary for the private native ABI.
pub(all) suberror MonotonicClockNativeError {
  NativeUnavailable
  NativeOutOfRange
} derive(Debug, Eq)

///|
pub extend MonotonicClockNativeError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend MonotonicClockNativeError with Eq::{not_equal, equal}

///|
fn finish_native_read(
  status : Int32,
  value : UInt64,
) -> UInt64 raise MonotonicClockNativeError {
  match status {
    NATIVE_OK => value
    NATIVE_UNAVAILABLE => raise NativeUnavailable
    NATIVE_OUT_OF_RANGE => raise NativeOutOfRange
    _ => raise NativeUnavailable
  }
}

///|
/// Reads milliseconds from the operating system's monotonic clock.
pub fn now_millis() -> UInt64 raise MonotonicClockNativeError {
  let output = Ref(0UL)
  let status = raw_now_millis(output)
  finish_native_read(status, output.val)
}

///|
/// Allocation-free read into caller-preallocated scalar storage.
pub fn now_millis_status(output : Ref[UInt64]) -> Int32 {
  raw_now_millis(output)
}
