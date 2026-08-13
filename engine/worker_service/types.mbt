///|
const FLIGHT_EMPTY : Int = 0

///|
const FLIGHT_PLANNED : Int = 1

///|
const FLIGHT_SUBMITTED : Int = 2

///|
const FLIGHT_RECEIVED : Int = 3

///|
const SERVICE_MODE_FIXTURE_RAW : Int = 0

///|
const SERVICE_MODE_ONLINE_ELIGIBLE : Int = 1

///|
const SERVICE_MODE_ONLINE_RETIRED : Int = 2

///|
const SERVICE_MODE_ONLINE_ACTIVE : Int = 3

///|
const ONLINE_REQUEST_EMPTY : Int = 0

///|
const ONLINE_REQUEST_LIVE : Int = 1

///|
const ONLINE_REQUEST_TERMINATION_PENDING : Int = 2

///|
const ONLINE_REQUEST_TERMINAL : Int = 3

///|
const ONLINE_REQUEST_RETIRED : Int = 4

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
  OwnershipMode
  Clock
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
  priv mut online : OnlineWorkerLease?
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
  priv mut ownership_mode : Int
  priv mut online_epoch : UInt64
  priv mut online_publication_cursor : UInt64
}

///|
/// Exclusive, epoch-authenticated access to the production-owned online mode.
/// The lease exposes no scheduler, process, request handle, request identity,
/// generation, or raw publication owner and deliberately has no `Debug`.
pub struct OnlineWorkerLease {
  priv service : WorkerService
  priv mut epoch : UInt64
  priv mut clock : @monotonic_clock.MonotonicClock
  priv mut request_phase : Int
  priv mut handle : @core.RequestHandle?
  priv mut request_generation : @inference.RequestGeneration
  priv mut cut_source_generation : @inference.RequestGeneration
  priv mut publication_cursor : UInt64
  priv mut expected_position : Int
  priv mut cancel_reserved : Bool
}

///|
#valtype
pub(all) enum OnlineWorkerPublicationKind {
  OnlineGeneratedTokenPublication
  OnlineSuppressedTokenPublication
  OnlineTerminalPublication
} derive(Debug, Eq)

///|
#valtype
pub(all) enum OnlineWorkerTerminalReason {
  OnlineCancelledByCaller
  OnlineDeadlineReached
  OnlineStopTokenReached
  OnlineMaximumOutputReached
  OnlineWorkerFailed
} derive(Debug, Eq)

///|
/// Sanitized scheduler publication. Request identity and generation remain in
/// the lease and cannot be paired independently by a caller.
#valtype
pub struct OnlineWorkerPublication(Int, Int, Int, UInt64)

///|
#valtype
pub(all) enum OnlineCancelReservationStart {
  OnlineCancelReserved
  OnlineCancelNaturalTerminal
} derive(Debug, Eq)

///|
#valtype
pub(all) enum OnlineCancelReservationCommit {
  OnlineCancelCommitted
  OnlineCancelReservationAbsent
} derive(Debug, Eq)

///|
#valtype
pub(all) enum OnlineExpiry {
  OnlineDeadlinePending
  OnlineDeadlineNaturalTerminal
  OnlineDeadlineCommitted
} derive(Debug, Eq)

///|
#valtype
priv enum OnlineExpiryClockPreflight {
  ExpiryNaturalTerminal
  ExpiryClockSample(UInt64)
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
