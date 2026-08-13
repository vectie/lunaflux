///|
const FLIGHT_EMPTY : Int = 0

///|
const FLIGHT_PLANNED : Int = 1

///|
const FLIGHT_SUBMITTED : Int = 2

///|
const FLIGHT_RECEIVED : Int = 3

///|
pub(all) enum WorkerServiceLifecycle {
  ServiceReady
  ServiceRecovering
  ServiceClosingFailed
  ServiceClosed
} derive(Debug, Eq)

///|
pub(all) enum WorkerServiceRule {
  Identity
  BootstrapIdentity
  BootstrapSourceIdentity
  DeviceAssignment
  Generation
  WorkerEnvelope
  InferenceEnvelope
  Sequence
  Lifecycle
  Recovery
} derive(Debug, Eq)

///|
pub(all) suberror WorkerServiceError {
  Invalid(WorkerServiceRule)
  Process(@worker_process.WorkerProcessError)
  Scheduler(@core.SchedulerError)
} derive(Debug, Eq)

///|
/// Allocation-free result of one scheduler-to-worker submission attempt.
#valtype
pub enum WorkerServiceDispatch {
  DispatchIdle
  DispatchBackpressured
  DispatchSubmitted(UInt64)
}

///|
pub(all) enum WorkerServiceRestart {
  Restarted
  RestartCleanupRequired(@worker_process.FailedWorkerProcessStartup)
}

///|
/// Independent expected deployment binding for one worker-service owner.
/// Scheduler identity, generation, sequence, and worker limits remain owned by
/// the scheduler and are deliberately not duplicated here.
pub struct WorkerServiceBinding {
  priv bootstrap_digest : @worker_wire.WorkerBootstrapDigest
  priv bootstrap_source_digest : @worker_wire.WorkerBootstrapSourceDigest
  priv device_ordinal : Int
  priv inference_limits : @inference.InferenceLimits
} derive(Debug, Eq)

///|
/// Thread-confined owner joining one scheduler to one isolated worker process.
pub struct WorkerService {
  priv scheduler : @core.Scheduler
  priv binding : WorkerServiceBinding
  priv mut process : @worker_process.WorkerProcessSupervisor
  priv flight_states : FixedArray[Int]
  priv flight_sequences : FixedArray[UInt64]
  priv mut lifecycle : WorkerServiceLifecycle
}

///|
pub fn WorkerServiceBinding::new(
  bootstrap_digest~ : @worker_wire.WorkerBootstrapDigest,
  bootstrap_source_digest~ : @worker_wire.WorkerBootstrapSourceDigest,
  device_ordinal~ : Int,
  inference_limits~ : @inference.InferenceLimits,
) -> WorkerServiceBinding raise WorkerServiceError {
  if device_ordinal < 0 || device_ordinal > 1023 {
    raise Invalid(DeviceAssignment)
  }
  {
    bootstrap_digest,
    bootstrap_source_digest,
    device_ordinal,
    inference_limits,
  }
}

///|
pub fn WorkerServiceBinding::bootstrap_source_digest(
  self : WorkerServiceBinding,
) -> @worker_wire.WorkerBootstrapSourceDigest {
  self.bootstrap_source_digest
}

///|
pub fn WorkerServiceBinding::bootstrap_digest(
  self : WorkerServiceBinding,
) -> @worker_wire.WorkerBootstrapDigest {
  self.bootstrap_digest
}

///|
pub fn WorkerServiceBinding::device_ordinal(self : WorkerServiceBinding) -> Int {
  self.device_ordinal
}

///|
pub fn WorkerServiceBinding::inference_limits(
  self : WorkerServiceBinding,
) -> @inference.InferenceLimits {
  self.inference_limits
}

///|
pub fn WorkerServiceDispatch::is_idle(self : WorkerServiceDispatch) -> Bool {
  self is DispatchIdle
}

///|
pub fn WorkerServiceDispatch::is_backpressured(
  self : WorkerServiceDispatch,
) -> Bool {
  self is DispatchBackpressured
}

///|
pub fn WorkerServiceDispatch::is_submitted(
  self : WorkerServiceDispatch,
) -> Bool {
  self is DispatchSubmitted(_)
}

///|
pub fn WorkerServiceDispatch::sequence_value(
  self : WorkerServiceDispatch,
) -> UInt64 raise WorkerServiceError {
  match self {
    DispatchSubmitted(sequence) => sequence
    _ => raise Invalid(Sequence)
  }
}
