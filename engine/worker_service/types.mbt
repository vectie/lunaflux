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
  ServiceDeviceInvalidationRequired
  ServiceRestartReady
  ServiceRestartForbidden
  ServiceTerminalCloseReady
  ServiceReplacementCleanup
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
  RootBoundProcess(@worker_process.RootBoundWorkerProcessError)
  Scheduler(@core.SchedulerError)
} derive(Debug, Eq)

///|
/// Payload-safe owned-instance preparation evidence.
pub(all) enum OwnedWorkerServiceFailure {
  OwnedSchedulerFailure(@core.SchedulerError)
  OwnedProcessFailure(@worker_process.RootBoundWorkerProcessError)
  OwnedServiceFailure(WorkerServiceError)
} derive(Debug, Eq)

///|
pub(all) suberror OwnedWorkerServicePreparationError {
  OwnedPreparationFailure(OwnedWorkerServiceFailure)
  OwnedPreparationCompoundFailure(
    OwnedWorkerServiceFailure,
    OwnedWorkerServiceFailure
  )
} derive(Debug, Eq)

///|
/// Cleanup-only owner returned when rooted startup or the final service join
/// could not consume every child/root authority.
pub struct FailedOwnedWorkerServicePreparation {
  priv mut initial : @worker_process.FailedRootBoundWorkerProcessStartup?
  priv mut ready_process : @worker_process.RootBoundWorkerProcessSupervisor?
  priv mut cause_kind : Int
  priv service_cause : WorkerServiceError?
  priv process_cause : @worker_process.RootBoundWorkerProcessError?
  priv mut cleanup_process_failure : @worker_process.RootBoundWorkerProcessError?
  priv mut closed : Bool
  priv mut reported_consumed_failure : Bool
}

///|
pub(all) enum OwnedWorkerServicePreparationState {
  OwnedServiceReady
  OwnedServiceCleanupRequired
} derive(Debug, Eq)

///|
/// Preallocated publication shell for owned preparation. The shell itself is
/// allocated before root acquisition; publishing either terminal state only
/// mutates its private tag.
pub struct OwnedWorkerServicePreparation {
  priv mut service : WorkerService?
  priv cleanup : FailedOwnedWorkerServicePreparation
  priv mut state : OwnedWorkerServicePreparationState
  priv mut consumed : Bool
}

///|
/// Allocation-free result of one owner-resident service transition.
#valtype
pub enum WorkerServiceProgress {
  ServiceIdle
  ServiceBackpressured
  ServiceStarted(UInt64)
  ServicePending(UInt64)
  ServiceCompletionReady(UInt64)
  ServiceCommitted(UInt64)
}

///|
pub(all) enum WorkerServiceRestart {
  Restarted
  RestartCleanupRequired
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
  priv mut process : @worker_process.RootBoundWorkerProcessSupervisor?
  priv mut flight_state : Int
  priv mut flight_sequence : UInt64
  priv mut lifecycle : WorkerServiceLifecycle
  priv production_owned : Bool
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
pub fn WorkerServiceProgress::sequence_value(
  self : WorkerServiceProgress,
) -> UInt64 raise WorkerServiceError {
  match self {
    ServiceStarted(sequence)
    | ServicePending(sequence)
    | ServiceCompletionReady(sequence)
    | ServiceCommitted(sequence) => sequence
    _ => raise Invalid(Sequence)
  }
}
