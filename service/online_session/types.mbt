///|
pub(all) enum OnlineSessionRule {
  Lifecycle
  Ownership
  Streaming
  InferenceEnvelope
  ModelIdentity
  EventCredit
} derive(Debug, Eq)

///|
pub(all) enum OnlineSessionFailure {
  AdmissionFailure
  ServicePreparationFailure
  ServiceFailure
  WireFailure
  Invalid(OnlineSessionRule)
} derive(Debug, Eq)

///|
pub(all) suberror OnlineSessionPreparationError {
  PreparationFailed(OnlineSessionFailure)
} derive(Debug, Eq)

///|
pub(all) suberror OnlineSessionError {
  SessionFailed(OnlineSessionFailure)
} derive(Debug, Eq)

///|
pub(all) enum OnlineSessionPreparationState {
  OnlineSessionReady
  OnlineSessionCleanupRequired
} derive(Debug, Eq)

///|
#valtype
pub(all) enum OnlineSessionCleanupProgress {
  OnlineSessionCleanupPending
  OnlineSessionCleanupClosed
} derive(Debug, Eq)

///|
pub struct OnlineSession {
  priv admitted : @request_admission.AdmittedRequest
  priv mut lease : @worker_service.OnlineWorkerLease?
  priv writer : @framed_wire.CanonicalEventWriter
  priv output : FixedArray[Byte]
  priv input_tokens : Int
  priv max_new_tokens : Int
  priv mut event_ready : Bool
  priv mut accepted_acked : Bool
  priv mut cleanup_phase : Int
  priv mut recovery_close : Bool
  priv mut closed : Bool
}

///|
pub struct FailedOnlineSessionPreparation {
  priv mut lease : @worker_service.OnlineWorkerLease?
  priv mut service_cleanup : @worker_service.FailedOwnedWorkerServicePreparation?
  priv mut cause_kind : Int
  priv mut closed : Bool
}

///|
pub struct OnlineSessionPreparation {
  priv mut session : OnlineSession?
  priv cleanup : FailedOnlineSessionPreparation
  priv mut state : OnlineSessionPreparationState
  priv mut consumed : Bool
}
