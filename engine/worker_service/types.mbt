///|
/// Exact owner-resident lifecycle for one physical worker exchange. Once a
/// completion is accepted, the scheduler exclusively owns its submitted epoch.
#valtype
priv enum WorkerFlightState {
  FlightEmpty
  FlightPlanned(UInt64)
  FlightSubmitted(UInt64)
  FlightReceived(UInt64)
  FlightAccepted(UInt64)
  FlightPendingFailure(UInt64, UInt64, Bool)
}

///|
/// Permanent ownership family selected for one worker-service owner.
#valtype
priv enum WorkerServiceOwnership {
  FixtureRawOwnership
  OnlineEligibleOwnership
  OnlineRetiredOwnership
  OnlineActiveOwnership
} derive(Eq)

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
  RestartBackoff
  WorkerOutput
} derive(Debug, Eq)

///|
/// Bounded backend-neutral physical failure. Exact process/group cleanup
/// authority remains inside the private transport or owned preparation shell.
pub(all) enum WorkerPhysicalTransportFailure {
  PhysicalBackpressured
  PhysicalInvalidOutput
  PhysicalUnavailable
  PhysicalCleanupFailed
  PhysicalLifecycleViolation
} derive(Debug, Eq)

///|
pub(all) suberror WorkerServiceError {
  Invalid(WorkerServiceRule)
  PhysicalTransport(WorkerPhysicalTransportFailure)
  Scheduler(@core.SchedulerError)
} derive(Debug, Eq)

///|
/// Payload-safe owned-instance preparation evidence.
pub(all) enum OwnedWorkerServiceFailure {
  OwnedSchedulerFailure(@core.SchedulerError)
  OwnedPhysicalTransportFailure(WorkerPhysicalTransportFailure)
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
  priv mut group : @tensor_parallel_group_transport.TensorParallelGroupTransport?
  priv mut cause_kind : Int
  priv mut service_cause : WorkerServiceError?
  priv process_cause : @worker_process.RootBoundWorkerProcessError?
  priv mut cleanup_process_failure : @worker_process.RootBoundWorkerProcessError?
  priv mut cleanup_group_failure : Bool
  priv mut closed : Bool
  priv mut reported_consumed_failure : Bool
}

///|
const MAINTENANCE_NONE : Int = 0

///|
const MAINTENANCE_RECOVERY : Int = 1

///|
const MAINTENANCE_SHUTDOWN : Int = 2

///|
const MAINTENANCE_REPLACEMENT_CLEANUP : Int = 3

///|
const MAINTENANCE_CHILD : Int = 1

///|
const MAINTENANCE_RETIRE : Int = 2

///|
const MAINTENANCE_INVALIDATE : Int = 3

///|
const MAINTENANCE_DRAIN : Int = 4

///|
const MAINTENANCE_BEGIN_CLOSE : Int = 5

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
  ServiceAdvanced
  ServiceBackpressured
  ServicePlanPending(UInt64)
  ServiceStarted(UInt64)
  ServicePending(UInt64)
  ServiceCompletionReady(UInt64)
  ServiceCommitted(UInt64)
}

///|
/// Allocation-free disposition of one cooperative owner maintenance step.
/// No process, root, scheduler-plan, request, or exit authority is exposed.
#valtype
pub(all) enum LunaWorkerServiceMaintenanceProgress {
  LunaWorkerServiceMaintenancePending
  LunaWorkerServiceMaintenanceAdvanced
  LunaWorkerServiceMaintenanceCleanupRequired
  LunaWorkerServiceMaintenanceCleanupStuck
  LunaWorkerServiceRecoveryReady
  LunaWorkerServiceFailCloseReady
  LunaWorkerServiceRestartReady
  LunaWorkerServiceClosed
} derive(Debug, Eq)

///|
/// Payload-safe classification of one online worker step. Lower scheduler,
/// protocol, process, and transport errors remain owned by the service.
#valtype
pub(all) enum OnlineWorkerStep {
  OnlineWorkerAdvanced
  OnlineWorkerRetryable
  OnlineWorkerOutputInvalid
  OnlineWorkerUnavailable
} derive(Debug, Eq)

///|
pub(all) enum WorkerServiceRestart {
  Restarted
  RestartCleanupRequired
}

///|
/// Immutable host-side crash-loop policy. It owns no clock, process, scheduler,
/// request, or cleanup authority.
pub struct WorkerRestartBackoffPolicy {
  priv initial_delay_millis : UInt64
  priv maximum_delay_millis : UInt64
  priv stable_success_millis : UInt64
  priv maximum_attempts : Int
} derive(Debug, Eq)

///|
/// Minimal cooperative replacement disposition consumed by online_session.
/// Restart generation and attempt authentication remain private to this
/// package.
#valtype
pub(all) enum OnlineWorkerRestartProgress {
  OnlineWorkerRestartWaiting(Int)
  OnlineWorkerRestarted
  OnlineWorkerRestartCleanupRequired
  OnlineWorkerRestartForbidden
} derive(Debug, Eq)

///|
priv enum WorkerPhysicalBinding {
  RootedSingleBinding(WorkerServiceBinding)
  TensorParallelBinding
}

///|
/// Instance-owned graph aggregation. Child-local counters are authenticated
/// before they are folded into the replacement-stable aggregate.
priv struct WorkerServiceGraphTelemetryOwner {
  aggregate : @worker_wire.WorkerGraphTelemetrySlot
  mut child_hits : UInt64
  mut child_misses : UInt64
  mut child_initialized : Bool
}

///|
/// Thread-confined owner joining one scheduler to one isolated worker process.
pub struct WorkerService {
  priv scheduler : @core.Scheduler
  priv identity : WorkerServiceIdentity
  priv physical_binding : WorkerPhysicalBinding
  priv mut transport : WorkerPhysicalTransport
  priv mut flight : WorkerFlightState
  priv mut pending_plan_sequence : UInt64
  priv mut maintenance_mode : Int
  priv mut maintenance_phase : Int
  priv mut maintenance_failure : @worker_protocol.WorkerFailure
  priv mut lifecycle : WorkerServiceLifecycle
  priv production_owned : Bool
  priv mut ownership_mode : WorkerServiceOwnership
  priv mut online_epoch : UInt64
  priv mut online_publication_cursor : UInt64
  priv graph_telemetry : WorkerServiceGraphTelemetryOwner
}

///|
/// Exclusive, epoch-authenticated access to the production-owned online mode.
/// The lease exposes no scheduler, process, request handle, request identity,
/// generation, or raw publication owner and deliberately has no `Debug`.
pub struct OnlineWorkerLease {
  priv service : WorkerService
  priv mut epoch : UInt64
  priv mut clock : @monotonic_clock.MonotonicClock
  priv restart_policy : WorkerRestartBackoffPolicy
  priv mut restart_attempts : Int
  priv mut restart_generation : UInt64
  priv mut restart_phase : Int
  priv mut restart_delay_millis : UInt64
  priv mut restart_deadline_millis : UInt64
  priv mut restart_cached_remaining_millis : Int
  priv mut restart_stability_generation : UInt64
  priv mut restart_stability_deadline_millis : UInt64
  priv mut restart_stability_predecessor : UInt64
  priv mut restart_clock_initialized : Bool
  priv mut restart_last_clock_millis : UInt64
  priv mut restart_forbidden : Bool
  priv mut prepared_admission : @core.PreparedExclusiveAdmission?
  priv prepared_clock_read : @monotonic_clock.PreparedMonotonicRead
  priv lane_phases : FixedArray[Int]
  priv lane_generations : FixedArray[UInt64]
  priv lane_handles : FixedArray[Array[@core.RequestHandle]]
  priv lane_semantic_retentions : FixedArray[
    @inference.LunaRequestStopTokenRetentionSlot,
  ]
  priv lane_request_generations : FixedArray[@inference.RequestGeneration]
  priv lane_cut_source_generations : FixedArray[@inference.RequestGeneration]
  // 0 none, 1 exact waiting request survived invalidation, 2 an exact
  // WorkerFailed terminal is already pending for post-restart retirement.
  priv lane_restart_dispositions : FixedArray[Int]
  priv lane_expected_positions : FixedArray[Int]
  priv free_lanes : FixedArray[Int]
  priv mut free_head : Int
  priv mut free_count : Int
  priv mut active_lane_count : Int
  // Compatibility methods authenticate one exact fixed lane and reject while
  // more than that lane is live. Production aggregates use request handles.
  priv mut compatibility_lane : Int
  priv mut compatibility_generation : UInt64
  priv mut publication_cursor : UInt64
  priv mut cancel_lane : Int
  priv mut cancel_generation : UInt64
}

///|
/// Exact generation-authenticated request lane retained by one online worker.
/// It exposes no scheduler handle, request identity, stop storage, or raw lane
/// epoch and deliberately has no `Debug`.
#valtype
pub struct LunaOnlineWorkerRequest {
  priv lease : OnlineWorkerLease
  priv lease_epoch : UInt64
  priv lane : Int
  priv generation : UInt64
}

///|
#valtype
pub(all) enum LunaOnlineWorkerAdmissionKind {
  LunaOnlineWorkerAdmitted
  LunaOnlineWorkerSaturated
  LunaOnlineWorkerDraining
} derive(Debug, Eq)

///|
/// Allocation-free admission disposition. The exact request-lane capability
/// can be extracted only from an admitted result.
#valtype
pub struct LunaOnlineWorkerAdmission {
  priv disposition : Int
  priv lease : OnlineWorkerLease
  priv lease_epoch : UInt64
  priv lane : Int
  priv generation : UInt64
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
/// Sanitized scheduler publication. Request identity and scheduler generation
/// remain private. Its route is the caller-owned scalar copied at admission;
/// it grants no scheduler or lane mutation authority.
#valtype
pub struct OnlineWorkerPublication(Int, Int, Int, UInt64, Int, UInt64)

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
pub(all) enum OnlineTerminalRecoveryProgress {
  OnlineTerminalRecoveryPending
  OnlineTerminalRecoveryPublicationReady
} derive(Debug, Eq)

///|
#valtype
pub(all) enum OnlineTerminalRecoveryStatus {
  OnlineTerminalRecoveryRetryable
  OnlineTerminalRecoveryStillPending
  OnlineTerminalRecoveryReady
} derive(Debug, Eq)

///|
#valtype
pub(all) enum OnlinePreparedAdmissionCommit {
  OnlineAdmissionCommitted
  OnlineAdmissionClockFailed
  OnlineAdmissionExpired
  OnlineAdmissionInvalidated
} derive(Debug, Eq)

///|
#valtype
priv enum OnlineExpiryClockPreflight {
  ExpiryNaturalTerminal
  ExpiryClockSample(UInt64)
}

///|
pub fn WorkerServiceProgress::sequence_value(
  self : WorkerServiceProgress,
) -> UInt64 raise WorkerServiceError {
  match self {
    ServicePlanPending(sequence)
    | ServiceStarted(sequence)
    | ServicePending(sequence)
    | ServiceCompletionReady(sequence)
    | ServiceCommitted(sequence) => sequence
    _ => raise Invalid(Sequence)
  }
}
