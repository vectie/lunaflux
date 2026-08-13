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
  priv mut next_transfer_epoch : UInt64
  priv mut active_write_epoch : UInt64
  priv mut active_read_epoch : UInt64
  priv mut failed : Bool
  priv mut closed : Bool
}

///|
/// Opaque authority for one frame write bound to its creating child and
/// transfer epoch. A child admits only one pending transfer in either
/// direction. The source must not be mutated until completion.
pub struct PendingFrameWrite {
  priv owner : ChildProcess
  priv epoch : UInt64
  priv source : FixedArray[Byte]
  priv source_offset : Int
  priv byte_count : Int
  priv deadline_millis : Int64
  priv mut phase : Int
  priv mut cursor : Int
}

///|
/// Opaque authority for one transactional frame read bound to its creating
/// child and transfer epoch. A child admits only one pending transfer in
/// either direction. Destination publication occurs only on success.
pub struct PendingFrameRead {
  priv owner : ChildProcess
  priv epoch : UInt64
  priv destination : FixedArray[Byte]
  priv staging : FixedArray[Byte]
  priv deadline_millis : Int64
  priv mut phase : Int
  priv mut cursor : Int
  priv mut byte_count : Int
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
