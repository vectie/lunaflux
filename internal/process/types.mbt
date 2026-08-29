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
  priv empty_transfer_buffer : FixedArray[Byte]
  priv mut write_transfer_active : Bool
  priv mut write_transfer_source : FixedArray[Byte]
  priv mut write_transfer_offset : Int
  priv mut write_transfer_byte_count : Int
  priv mut write_transfer_deadline_millis : Int64
  priv mut write_transfer_phase : Int
  priv mut write_transfer_cursor : Int
  priv mut read_transfer_active : Bool
  priv mut read_transfer_destination : FixedArray[Byte]
  priv mut read_transfer_staging : FixedArray[Byte]
  priv mut read_transfer_byte_count : Int
  priv mut read_transfer_deadline_millis : Int64
  priv mut read_transfer_phase : Int
  priv mut read_transfer_cursor : Int
  priv mut maintenance_phase : Int
  priv mut maintenance_phase_started_millis : Int64
  priv mut maintenance_deadline_millis : Int64
  priv mut maintenance_resume_phase : Int
  priv mut maintenance_exit_kind : Int
  priv mut maintenance_exit_code : Int
  priv mut maintenance_has_exit : Bool
  priv mut failed : Bool
  priv mut closed : Bool
}

///|
/// Allocation-complete child shell prepared before any external capability is
/// borrowed. Spawning only installs native pid/fd state into this exact owner.
pub struct PreparedChildProcess {
  priv child : ChildProcess
  priv approved_executable : @executable_capability.ApprovedExecutableHandle
  priv mut consumed : Bool
  priv mut cleanup_status : Int
}

///|
pub fn PreparedChildProcess::child_owner(
  self : PreparedChildProcess,
) -> ChildProcess {
  self.child
}

///|
/// Returns scalar cleanup evidence retained by a failed prepared spawn.
pub fn PreparedChildProcess::cleanup_status(self : PreparedChildProcess) -> Int {
  self.cleanup_status
}

///|
/// Worker-side owner of the inherited standard-input/output channel.
pub struct InheritedChannel {
  priv limits : ProcessLimits
  priv write_prefix : FixedArray[Byte]
  priv read_prefix : FixedArray[Byte]
  priv empty_transfer_buffer : FixedArray[Byte]
  priv mut write_transfer_active : Bool
  priv mut write_transfer_source : FixedArray[Byte]
  priv mut write_transfer_offset : Int
  priv mut write_transfer_byte_count : Int
  priv mut write_transfer_deadline_millis : Int64
  priv mut write_transfer_phase : Int
  priv mut write_transfer_cursor : Int
  priv mut read_transfer_active : Bool
  priv mut read_transfer_destination : FixedArray[Byte]
  priv mut read_transfer_staging : FixedArray[Byte]
  priv mut read_transfer_byte_count : Int
  priv mut read_transfer_deadline_millis : Int64
  priv mut read_transfer_phase : Int
  priv mut read_transfer_cursor : Int
  priv mut read_transfer_allows_eof : Bool
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
/// Scalar progress evidence for owner-resident cooperative child maintenance.
#valtype
pub struct ChildProcessMaintenanceProgress {
  priv kind : Int
}

///|
/// Scalar result of one nonblocking liveness observation while the child has
/// no active transfer or maintenance transition.
#valtype
pub struct ChildProcessIdleObservation {
  priv kind : Int
  priv exit_kind : Int
  priv exit_code : Int
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
pub fn ChildProcessMaintenanceProgress::is_pending(
  self : ChildProcessMaintenanceProgress,
) -> Bool {
  self.kind == 0
}

///|
pub fn ChildProcessMaintenanceProgress::is_advanced(
  self : ChildProcessMaintenanceProgress,
) -> Bool {
  self.kind == 1
}

///|
pub fn ChildProcessMaintenanceProgress::is_complete(
  self : ChildProcessMaintenanceProgress,
) -> Bool {
  self.kind == 2
}

///|
pub fn ChildProcessMaintenanceProgress::is_cleanup_required(
  self : ChildProcessMaintenanceProgress,
) -> Bool {
  self.kind == 3
}

///|
/// Returns true after the bounded final-reap interval expired while the exact
/// unreaped child owner remains retained for later nonblocking poll progress.
pub fn ChildProcessMaintenanceProgress::is_cleanup_stuck(
  self : ChildProcessMaintenanceProgress,
) -> Bool {
  self.kind == 4
}

///|
pub fn ChildProcessIdleObservation::is_alive(
  self : ChildProcessIdleObservation,
) -> Bool {
  self.kind == 0
}

///|
pub fn ChildProcessIdleObservation::has_exited(
  self : ChildProcessIdleObservation,
) -> Bool {
  self.kind == 1
}

///|
/// Returns the exact copied exit evidence after `has_exited` succeeds.
pub fn ChildProcessIdleObservation::exit(
  self : ChildProcessIdleObservation,
) -> ChildExit raise ProcessError {
  if self.kind != 1 {
    raise InvalidState(Wait)
  }
  { kind: self.exit_kind, code: self.exit_code }
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
