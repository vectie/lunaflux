///|
pub(all) enum WorkerProcessLifecycle {
  Ready
  Failed
  ClosingFailed
  Closed
} derive(Debug, Eq)

///|
pub(all) enum WorkerProcessRule {
  Capacity
  SourceMismatch
  Sequence
  Submission
  Completion
  Lifecycle
  Recovery
} derive(Debug, Eq)

///|
/// Payload-safe native transport failure vocabulary.
pub(all) enum WorkerTransportFailure {
  InvalidConfiguration
  InvalidState
  Timeout
  ChannelClosed
  OperationFailed
  SpawnFailed
} derive(Debug, Eq)

///|
pub(all) suberror WorkerProcessError {
  Invalid(WorkerProcessRule)
  Backpressured
  OwnerMismatch
  Transport(WorkerTransportFailure)
  Wire(@worker_wire.WorkerWireError)
  Protocol(@worker_protocol.WorkerProtocolError)
} derive(Debug, Eq)

///|
/// Public limits owned by the supervisor boundary, not its native ABI. Startup
/// I/O has a separate per-prefix/per-payload timeout from steady plan traffic.
pub struct WorkerProcessLimits {
  priv max_frame_bytes : Int
  priv startup_io_timeout_millis : Int
  priv io_timeout_millis : Int
  priv shutdown_timeout_millis : Int
}

///|
/// Single-owner A/B process supervisor.
pub struct WorkerProcessSupervisor {
  priv child : @process.ChildProcess
  priv startup : @worker_wire.WorkerStartupContract
  priv source : @worker_wire.EncodedBootstrapSource
  priv plan_a : @worker_wire.PlanFrameBuffer
  priv plan_b : @worker_wire.PlanFrameBuffer
  priv completion_a : @worker_wire.CompletionFrameBuffer
  priv completion_b : @worker_wire.CompletionFrameBuffer
  priv tx_a : FixedArray[Byte]
  priv tx_b : FixedArray[Byte]
  priv rx : FixedArray[Byte]
  priv rx_staging : FixedArray[Byte]
  priv exchange_request_ids : FixedArray[UInt64]
  priv exchange_request_generations : FixedArray[UInt64]
  priv exchange_expected_kinds : FixedArray[Int]
  priv exchange_processed_tokens : FixedArray[Int]
  priv exchange_slots : FixedArray[Int]
  priv mut exchange_entry_count : Int
  priv mut state_a : Int
  priv mut state_b : Int
  priv mut sequence_a : UInt64
  priv mut sequence_b : UInt64
  priv mut last_submitted : UInt64
  priv mut last_retired : UInt64
  priv mut lifecycle : WorkerProcessLifecycle
}

///|
/// Startup outcome preserving cleanup authority after a double failure.
pub(all) enum WorkerProcessPreparation {
  ReadyProcess(WorkerProcessSupervisor)
  CleanupRequired(FailedWorkerProcessStartup)
}

///|
/// Opaque retained child cleanup after startup and close both fail.
pub struct FailedWorkerProcessStartup {
  priv child : @process.ChildProcess
  priv cause : WorkerProcessError
  priv mut cleanup_failure : WorkerProcessError
  priv mut closed : Bool
}

///|
/// Exact submitted-side capability. No process or buffer owner escapes.
#valtype
pub struct WorkerSubmission(WorkerProcessSupervisor, Int, UInt64)

///|
/// Completion-frame retention capability held through scheduler acceptance.
#valtype
pub struct ReceivedWorkerCompletion(WorkerProcessSupervisor, Int, UInt64)

///|
pub fn WorkerProcessLimits::new(
  max_frame_bytes~ : Int,
  startup_io_timeout_millis~ : Int,
  io_timeout_millis~ : Int,
  shutdown_timeout_millis~ : Int,
) -> WorkerProcessLimits raise WorkerProcessError {
  if max_frame_bytes < 64 ||
    max_frame_bytes > 67108864 ||
    startup_io_timeout_millis <= 0 ||
    startup_io_timeout_millis > 600000 ||
    io_timeout_millis <= 0 ||
    io_timeout_millis > 600000 ||
    shutdown_timeout_millis <= 0 ||
    shutdown_timeout_millis > 600000 {
    raise Invalid(Capacity)
  }
  {
    max_frame_bytes,
    startup_io_timeout_millis,
    io_timeout_millis,
    shutdown_timeout_millis,
  }
}

///|
pub fn WorkerProcessLimits::startup_io_timeout_millis(
  self : WorkerProcessLimits,
) -> Int {
  self.startup_io_timeout_millis
}

///|
pub fn WorkerProcessLimits::max_frame_bytes(self : WorkerProcessLimits) -> Int {
  self.max_frame_bytes
}

///|
pub fn WorkerProcessLimits::io_timeout_millis(
  self : WorkerProcessLimits,
) -> Int {
  self.io_timeout_millis
}

///|
pub fn WorkerProcessLimits::shutdown_timeout_millis(
  self : WorkerProcessLimits,
) -> Int {
  self.shutdown_timeout_millis
}

///|
pub fn WorkerProcessSupervisor::lifecycle(
  self : WorkerProcessSupervisor,
) -> WorkerProcessLifecycle {
  self.lifecycle
}

///|
pub fn WorkerProcessSupervisor::startup_contract(
  self : WorkerProcessSupervisor,
) -> @worker_wire.WorkerStartupContract {
  self.startup
}

///|
/// Returns the immutable canonical bootstrap source retained for replacements.
pub fn WorkerProcessSupervisor::bootstrap_source(
  self : WorkerProcessSupervisor,
) -> @worker_wire.EncodedBootstrapSource {
  self.source
}

///|
pub fn FailedWorkerProcessStartup::cause(
  self : FailedWorkerProcessStartup,
) -> WorkerProcessError {
  self.cause
}

///|
pub fn FailedWorkerProcessStartup::cleanup_failure(
  self : FailedWorkerProcessStartup,
) -> WorkerProcessError {
  self.cleanup_failure
}

///|
pub fn FailedWorkerProcessStartup::is_closed(
  self : FailedWorkerProcessStartup,
) -> Bool {
  self.closed
}

///|
/// Retries exact child close/reap after startup cleanup failed.
pub fn FailedWorkerProcessStartup::retry_close(
  self : FailedWorkerProcessStartup,
) -> Unit raise WorkerProcessError {
  if self.closed {
    return
  }
  self.child.close() catch {
    error => {
      self.cleanup_failure = transport_error(error)
      raise self.cleanup_failure
    }
  }
  self.closed = true
}

///|
pub fn WorkerProcessSupervisor::has_capacity(
  self : WorkerProcessSupervisor,
) -> Bool {
  self.lifecycle == Ready && (self.state_a == 0 || self.state_b == 0)
}

///|
pub fn WorkerProcessSupervisor::outstanding_count(
  self : WorkerProcessSupervisor,
) -> Int {
  (if self.state_a == 0 { 0 } else { 1 }) +
  (if self.state_b == 0 { 0 } else { 1 })
}

///|
pub fn WorkerSubmission::sequence_value(self : WorkerSubmission) -> UInt64 {
  self.2
}

///|
pub fn ReceivedWorkerCompletion::sequence_value(
  self : ReceivedWorkerCompletion,
) -> UInt64 {
  self.2
}

///|
/// Publishes child EOF and waits for a clean zero exit before final close.
pub fn WorkerProcessSupervisor::shutdown_clean(
  self : WorkerProcessSupervisor,
) -> Unit raise WorkerProcessError {
  self.require_ready()
  if self.outstanding_count() != 0 {
    raise Invalid(Lifecycle)
  }
  self.child.shutdown_write() catch {
    error => {
      self.lifecycle = Failed
      raise transport_error(error)
    }
  }
  let exit = self.child.wait() catch {
    error => {
      self.lifecycle = Failed
      raise transport_error(error)
    }
  }
  if exit.kind() != 0 || exit.code() != 0 {
    self.lifecycle = Failed
    raise Transport(OperationFailed)
  }
}
