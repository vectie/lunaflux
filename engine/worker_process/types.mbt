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
  priv mut child_selected : Bool
  priv startup : @worker_wire.WorkerStartupContract
  priv source : @worker_wire.EncodedBootstrapSource
  priv parent_approval : @worker_wire.WorkerParentApprovalSubjects?
  priv plan_a : @worker_wire.PlanFrameBuffer
  priv plan_b : @worker_wire.PlanFrameBuffer
  priv completion_a : @worker_wire.CompletionFrameBuffer
  priv completion_b : @worker_wire.CompletionFrameBuffer
  priv tx_a : FixedArray[Byte]
  priv tx_b : FixedArray[Byte]
  priv rx : FixedArray[Byte]
  priv rx_staging : FixedArray[Byte]
  priv graph_rx : FixedArray[Byte]
  priv graph_rx_staging : FixedArray[Byte]
  priv mut state_a : Int
  priv mut state_b : Int
  priv mut sequence_a : UInt64
  priv mut sequence_b : UInt64
  priv graph_telemetry_a : @worker_wire.WorkerGraphTelemetrySlot
  priv graph_telemetry_b : @worker_wire.WorkerGraphTelemetrySlot
  priv last_graph_telemetry : @worker_wire.WorkerGraphTelemetrySlot
  priv mut last_submitted : UInt64
  priv mut last_retired : UInt64
  priv mut maintenance_phase : Int
  priv mut maintenance_failure : WorkerProcessError?
  priv mut lifecycle : WorkerProcessLifecycle
}

///|
/// Private bounded-buffer startup plan. Deterministic capacities and reusable
/// worker buffers, child ownership, and canonical Configure/source/expected-
/// Ready bytes are validated and allocated before root acquisition. Startup
/// I/O, exact Ready comparison, and cleanup use scalar status paths.
priv struct PreparedWorkerProcessStartup {
  approved_executable : @worker_executable_file.WorkerExecutableAdmission
  child : @process.PreparedChildProcess
  startup_outgoing : FixedArray[Byte]
  startup_outgoing_length : Int
  source_outgoing : FixedArray[Byte]
  attestation_outgoing : FixedArray[Byte]
  attestation_outgoing_length : Int
  startup_incoming : FixedArray[Byte]
  expected_ready : FixedArray[Byte]
  expected_ready_length : Int
  supervisor : WorkerProcessSupervisor
  failed : FailedWorkerProcessStartup
}

///|
/// Startup outcome preserving cleanup authority after a double failure.
#valtype
pub(all) enum WorkerProcessPreparation {
  ReadyProcess(WorkerProcessSupervisor)
  CleanupRequired(FailedWorkerProcessStartup)
}

///|
/// Opaque retained child cleanup after startup and close both fail.
pub struct FailedWorkerProcessStartup {
  priv child : @process.ChildProcess
  priv mut child_selected : Bool
  priv mut cause_kind : Int
  priv mut cause_status : Int
  priv mut cleanup_status : Int
  priv mut cleanup_active : Bool
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
/// Allocation-free scalar progress evidence for owner-resident child cleanup.
#valtype
pub struct WorkerProcessMaintenanceProgress {
  priv kind : Int
}

///|
/// Exact payload-safe exit evidence copied out of the private process owner.
#valtype
pub struct WorkerProcessExit {
  priv kind : Int
  priv code : Int
}

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
  startup_raw_cause(self)
}

///|
pub fn FailedWorkerProcessStartup::cleanup_failure(
  self : FailedWorkerProcessStartup,
) -> WorkerProcessError {
  if self.cleanup_status == 0 {
    Invalid(Lifecycle)
  } else {
    transport_status_error(self.cleanup_status)
  }
}

///|
pub fn FailedWorkerProcessStartup::is_closed(
  self : FailedWorkerProcessStartup,
) -> Bool {
  self.closed
}

///|
/// Retries exact child close/reap after startup cleanup retained authority.
/// A nonzero scalar close status is consumed evidence: the native authority
/// is already closed, so later calls are idempotent and preserve that status.
pub fn FailedWorkerProcessStartup::retry_close(
  self : FailedWorkerProcessStartup,
) -> Unit raise WorkerProcessError {
  if self.cleanup_active {
    raise Invalid(Lifecycle)
  }
  if self.closed {
    return
  }
  if !self.child_selected {
    abort("failed worker child selection invariant")
  }
  let child = self.child
  let status = child.startup_close_status()
  if status != 0 {
    self.cleanup_status = status
  }
  self.closed = child.is_closed()
  self.child_selected = !self.closed
  if status != 0 {
    raise transport_status_error(status)
  }
}

///|
fn FailedWorkerProcessStartup::begin_cleanup_maintenance(
  self : FailedWorkerProcessStartup,
) -> Unit raise WorkerProcessError {
  if self.closed || self.cleanup_active || !self.child_selected {
    raise Invalid(Lifecycle)
  }
  self.child.begin_shutdown_maintenance() catch {
    error => raise transport_error(error)
  }
  self.cleanup_active = true
}

///|
fn FailedWorkerProcessStartup::progress_cleanup_maintenance(
  self : FailedWorkerProcessStartup,
) -> WorkerProcessMaintenanceProgress raise WorkerProcessError {
  if !self.cleanup_active || self.closed || !self.child_selected {
    raise Invalid(Lifecycle)
  }
  let progress = self.child.progress_shutdown_maintenance() catch {
    error => {
      self.cleanup_status = 4
      raise transport_error(error)
    }
  }
  if progress.is_cleanup_required() {
    self.cleanup_status = 4
    return worker_maintenance_cleanup_required
  }
  if progress.is_cleanup_stuck() {
    return worker_maintenance_cleanup_stuck
  }
  if progress.is_complete() {
    self.closed = self.child.is_closed()
    self.child_selected = !self.closed
    self.cleanup_active = false
    if !self.closed {
      self.cleanup_status = 4
      return worker_maintenance_cleanup_required
    }
    return worker_maintenance_complete
  }
  if progress.is_advanced() {
    worker_maintenance_advanced
  } else {
    worker_maintenance_pending
  }
}

///|
fn FailedWorkerProcessStartup::cleanup_maintenance_remaining_millis(
  self : FailedWorkerProcessStartup,
) -> Int raise WorkerProcessError {
  if !self.cleanup_active || self.closed {
    raise Invalid(Lifecycle)
  }
  self.child.maintenance_remaining_millis() catch {
    error => raise transport_error(error)
  }
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
pub fn WorkerProcessMaintenanceProgress::is_pending(
  self : WorkerProcessMaintenanceProgress,
) -> Bool {
  self.kind == 0
}

///|
pub fn WorkerProcessMaintenanceProgress::is_advanced(
  self : WorkerProcessMaintenanceProgress,
) -> Bool {
  self.kind == 1
}

///|
pub fn WorkerProcessMaintenanceProgress::is_complete(
  self : WorkerProcessMaintenanceProgress,
) -> Bool {
  self.kind == 2
}

///|
pub fn WorkerProcessMaintenanceProgress::is_cleanup_required(
  self : WorkerProcessMaintenanceProgress,
) -> Bool {
  self.kind == 3
}

///|
/// Returns true when the bounded final-reap interval expired but the same
/// retained supervisor can still make later nonblocking reap progress.
pub fn WorkerProcessMaintenanceProgress::is_cleanup_stuck(
  self : WorkerProcessMaintenanceProgress,
) -> Bool {
  self.kind == 4
}

///|
pub fn WorkerProcessExit::kind(self : WorkerProcessExit) -> Int {
  self.kind
}

///|
pub fn WorkerProcessExit::code(self : WorkerProcessExit) -> Int {
  self.code
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
  self.require_child().shutdown_write() catch {
    error => {
      self.lifecycle = Failed
      raise transport_error(error)
    }
  }
  let exit = self.require_child().wait() catch {
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
