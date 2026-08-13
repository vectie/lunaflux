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
  Sequence
  Submission
  Completion
  Lifecycle
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
/// Public limits owned by the supervisor boundary, not its native ABI.
pub struct WorkerProcessLimits {
  priv max_frame_bytes : Int
  priv io_timeout_millis : Int
  priv shutdown_timeout_millis : Int
}

///|
/// Single-owner A/B process supervisor.
pub struct WorkerProcessSupervisor {
  priv child : @process.ChildProcess
  priv plan_a : @worker_wire.PlanFrameBuffer
  priv plan_b : @worker_wire.PlanFrameBuffer
  priv completion_a : @worker_wire.CompletionFrameBuffer
  priv completion_b : @worker_wire.CompletionFrameBuffer
  priv tx_a : FixedArray[Byte]
  priv tx_b : FixedArray[Byte]
  priv rx : FixedArray[Byte]
  priv mut state_a : Int
  priv mut state_b : Int
  priv mut sequence_a : UInt64
  priv mut sequence_b : UInt64
  priv mut last_submitted : UInt64
  priv mut last_retired : UInt64
  priv mut lifecycle : WorkerProcessLifecycle
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
  io_timeout_millis~ : Int,
  shutdown_timeout_millis~ : Int,
) -> WorkerProcessLimits raise WorkerProcessError {
  if max_frame_bytes < 64 ||
    max_frame_bytes > 67108864 ||
    io_timeout_millis <= 0 ||
    io_timeout_millis > 600000 ||
    shutdown_timeout_millis <= 0 ||
    shutdown_timeout_millis > 600000 {
    raise Invalid(Capacity)
  }
  { max_frame_bytes, io_timeout_millis, shutdown_timeout_millis }
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
