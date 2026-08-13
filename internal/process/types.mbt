///|
pub(all) enum ProcessLimitRule {
  FrameBytes
  IoTimeoutMillis
  ShutdownTimeoutMillis
} derive(Debug, Eq)

///|
pub(all) enum ProcessOperation {
  Write
  Read
  ShutdownWrite
  Wait
  Terminate
  Close
} derive(Debug, Eq)

///|
pub(all) suberror ProcessError {
  InvalidLimit(ProcessLimitRule)
  InvalidRange(ProcessOperation)
  InvalidState(ProcessOperation)
  SpawnFailed
  Timeout(ProcessOperation)
  ChannelClosed(ProcessOperation)
  OperationFailed(ProcessOperation)
} derive(Debug, Eq)

///|
/// Validated fixed bounds for one child channel.
pub struct ProcessLimits {
  priv max_frame_bytes : Int
  priv io_timeout_millis : Int
  priv shutdown_timeout_millis : Int
}

///|
/// Explicit owner of one authenticated socketpair child endpoint.
pub struct ChildProcess {
  priv handle : NativeChild
  priv limits : ProcessLimits
  priv write_prefix : FixedArray[Byte]
  priv read_prefix : FixedArray[Byte]
  priv mut transfer_kind : Int
  priv empty_transfer_buffer : FixedArray[Byte]
  priv mut transfer_primary : FixedArray[Byte]
  priv mut transfer_secondary : FixedArray[Byte]
  priv mut transfer_offset : Int
  priv mut transfer_byte_count : Int
  priv mut transfer_deadline_millis : Int64
  priv mut transfer_phase : Int
  priv mut transfer_cursor : Int
  priv mut failed : Bool
  priv mut closed : Bool
}

///|
/// Worker-side owner of the inherited standard-input/output channel.
pub struct InheritedChannel {
  priv limits : ProcessLimits
  priv write_prefix : FixedArray[Byte]
  priv read_prefix : FixedArray[Byte]
  priv mut failed : Bool
}

///|
/// Bounded child termination evidence.
#valtype
pub struct ChildExit {
  priv kind : Int
  priv code : Int
}

///|
pub fn ProcessLimits::new(
  max_frame_bytes~ : Int,
  io_timeout_millis~ : Int,
  shutdown_timeout_millis~ : Int,
) -> ProcessLimits raise ProcessError {
  if max_frame_bytes < 64 || max_frame_bytes > 67108864 {
    raise InvalidLimit(FrameBytes)
  }
  if io_timeout_millis <= 0 || io_timeout_millis > 600000 {
    raise InvalidLimit(IoTimeoutMillis)
  }
  if shutdown_timeout_millis <= 0 || shutdown_timeout_millis > 600000 {
    raise InvalidLimit(ShutdownTimeoutMillis)
  }
  { max_frame_bytes, io_timeout_millis, shutdown_timeout_millis }
}

///|
pub fn ProcessLimits::max_frame_bytes(self : ProcessLimits) -> Int {
  self.max_frame_bytes
}

///|
pub fn ProcessLimits::io_timeout_millis(self : ProcessLimits) -> Int {
  self.io_timeout_millis
}

///|
pub fn ProcessLimits::shutdown_timeout_millis(self : ProcessLimits) -> Int {
  self.shutdown_timeout_millis
}

///|
pub fn ChildExit::kind(self : ChildExit) -> Int {
  self.kind
}

///|
pub fn ChildExit::code(self : ChildExit) -> Int {
  self.code
}

///|
pub fn ChildProcess::is_closed(self : ChildProcess) -> Bool {
  self.closed
}

///|
pub fn ChildProcess::is_failed(self : ChildProcess) -> Bool {
  self.failed
}

///|
pub fn InheritedChannel::is_failed(self : InheritedChannel) -> Bool {
  self.failed
}
