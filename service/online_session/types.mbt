///|
pub(all) enum LunaOnlineInstanceRule {
  LunaOnlineLifecycle
  LunaOnlineOwnership
  LunaOnlineStreaming
  LunaOnlineInferenceEnvelope
  LunaOnlineModelIdentity
  LunaOnlineEventCredit
  LunaOnlineRequestEpoch
} derive(Debug, Eq)

///|
pub(all) enum LunaOnlineInstanceFailure {
  LunaOnlineAdmissionFailure
  LunaOnlineServicePreparationFailure
  LunaOnlineServiceFailure
  LunaOnlineInvalid(LunaOnlineInstanceRule)
} derive(Debug, Eq)

///|
pub(all) suberror LunaOnlineInstancePreparationError {
  LunaOnlineInstancePreparationFailed(LunaOnlineInstanceFailure)
} derive(Debug, Eq)

///|
pub(all) suberror LunaOnlineInstanceError {
  LunaOnlineInstanceFailed(LunaOnlineInstanceFailure)
} derive(Debug, Eq)

///|
pub(all) enum LunaOnlineInstancePreparationState {
  LunaOnlineInstanceReady
  LunaOnlineInstanceCleanupRequired
} derive(Debug, Eq)

///|
#valtype
pub struct LunaOnlineRequestTicket {
  priv lane : Int
  priv epoch : UInt64
} derive(Eq)

///|
/// One exact request/event-epoch capability. It grants semantic read access
/// and the sole acknowledgement authority for its pinned event.
#valtype
pub struct LunaOnlineEventCredit {
  priv instance : LunaOnlineInstance
  priv lane : Int
  priv request_epoch : UInt64
  priv event_epoch : UInt64
}

///|
#valtype
pub(all) enum LunaOnlineInstanceAdmissionKind {
  LunaOnlineInstanceAdmitted
  LunaOnlineInstanceBusy
  LunaOnlineInstanceDraining
} derive(Debug, Eq)

///|
/// Allocation-free admission disposition. Its ticket epoch remains private
/// and can be extracted only from an admitted result.
#valtype
pub struct LunaOnlineInstanceAdmission {
  priv disposition : Int
  priv lane : Int
  priv request_epoch : UInt64
}

///|
#valtype
pub(all) enum LunaOnlineInstanceProgress {
  LunaOnlineInstancePending
  LunaOnlineInstanceWaiting
  LunaOnlineInstanceEventReady
  LunaOnlineInstanceTerminalizationRequired
  LunaOnlineInstanceRequestRetirementRequired
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaOnlineInstanceRetirementProgress {
  LunaOnlineInstanceRetirementPending
  LunaOnlineInstanceIdle
  LunaOnlineInstanceClosed
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaOnlineInstanceShutdownProgress {
  LunaOnlineInstanceShutdownPending
  LunaOnlineInstanceShutdownClosed
} derive(Debug, Eq)

///|
#valtype
#warnings("-unused_constructor")
priv enum SessionEventKind {
  // Reserved so the established generated scalar event codes remain 1..6.
  NoSessionEvent
  AcceptedEvent
  TokenEvent
  CompletionUsageEvent
  CompletedEvent
  FailureUsageEvent
  FailedEvent
}

///|
#valtype
priv enum SessionFinishKind {
  FinishPending
  FinishStopSequence
  FinishTokenLimit
  FinishCancelled
}

///|
#valtype
priv enum SessionStringCut {
  NoStringCut
  CancelledStringCut
  NaturalStringCut
} derive(Eq)

///|
#valtype
priv enum SessionTerminalCut {
  NoTerminalCut
  CallerCancellationCut
  DeadlineFailureCut
  OutputCancellationCut
  WorkerFailureCut
  NaturalOutputFailureCut
} derive(Eq)

///|
#valtype
priv enum SessionLatchedFailure {
  NoLatchedFailure
  DeadlineLatchedFailure
  OutputLatchedFailure
  WorkerLatchedFailure
} derive(Eq)

///|
#valtype
priv enum LunaInstancePhase {
  LunaInstanceIdlePhase
  LunaRequestActivePhase
  LunaRequestAwaitingTerminalPhase
  LunaRequestReleaseReadyPhase
  LunaInstanceRecoveringTerminalPhase
  LunaInstanceClosingTerminalPhase
  LunaInstanceRetryingTerminalClosePhase
  LunaInstanceDrainingPhase
  LunaInstanceRetryingEmptyClosePhase
  LunaInstanceClosedPhase
} derive(Eq)

///|
#valtype
priv enum LunaRecoveryMaintenanceStep {
  LunaRecoveryMaintenancePending
  LunaRecoveryMaintenanceRestarted
  LunaRecoveryMaintenanceFailCloseReady
  LunaRecoveryMaintenanceClosed
}

///|
#valtype
priv enum PreparationFailureKind {
  PreparationOwnershipFailure
  PreparationServiceCleanupFailure
}

///|
#valtype
priv enum ExpiryCutClassification {
  ExpiryCutPending
  ExpiryCutNaturalTerminal
  ExpiryCutCommitted
}

///|
pub struct LunaOnlineInstance {
  priv expected_tokenizer : @tokenizer.TokenizerDigest
  priv expected_model : @spec.ModelIdentity
  priv inference : @inference.InferenceLimits
  priv mut claim : @request_admission.LunaPreparedRequestClaim?
  priv mut lease : @worker_service.OnlineWorkerLease?
  priv events : @luna_event.LunaEventOwner
  priv output : FixedArray[Byte]
  priv mut input_tokens : Int
  priv mut cached_input_tokens : Int
  priv mut request_epoch : UInt64
  priv mut event_epoch : UInt64
  priv mut event_kind : SessionEventKind
  priv mut event_taken : Bool
  priv mut accepted_acked : Bool
  priv mut output_tokens : Int
  priv mut final_tail_length : Int
  priv mut finish_kind : SessionFinishKind
  priv mut string_cut : SessionStringCut
  priv mut terminal_cut : SessionTerminalCut
  priv mut deferred_cancel : Bool
  priv deadline_code : FixedArray[Byte]
  priv output_code : FixedArray[Byte]
  priv worker_code : FixedArray[Byte]
  priv mut failure_kind : SessionLatchedFailure
  priv mut phase : LunaInstancePhase
  priv mut maintenance_state : Int
  priv mut restart_observation_pending : Bool
  priv mut restarted_failure_terminal_pending : Bool
  priv mut drain_requested : Bool
  priv mut recovery_close : Bool
  priv multi : LunaOnlineMultiState
}

///|
pub struct FailedLunaOnlineInstancePreparation {
  priv mut service_cleanup : @worker_service.FailedOwnedWorkerServicePreparation?
  priv mut cause_kind : PreparationFailureKind
  priv mut closed : Bool
}

///|
pub struct LunaOnlineInstancePreparation {
  priv mut instance : LunaOnlineInstance?
  priv cleanup : FailedLunaOnlineInstancePreparation
  priv mut state : LunaOnlineInstancePreparationState
  priv mut consumed : Bool
}
