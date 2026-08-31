///|
pub(all) enum LunaOnlineFramedCoordinatorRule {
  LunaOnlineFramedLifecycle
  LunaOnlineFramedIngress
  LunaOnlineFramedPreparation
  LunaOnlineFramedEvent
  LunaOnlineFramedOutputStall
  LunaOnlineFramedClock
  LunaOnlineFramedMaintenance
  LunaOnlineFramedOwnership
} derive(Debug, Eq)

///|
pub(all) suberror LunaOnlineFramedCoordinatorError {
  LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedCoordinatorRule)
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaOnlineFramedIngressKind {
  LunaOnlineFramedIngressAccepted
  LunaOnlineFramedIngressBackpressured
  LunaOnlineFramedIngressDraining
} derive(Debug, Eq)

///|
#valtype
pub struct LunaOnlineFramedIngress {
  priv disposition : Int
  priv consumed : Int
  priv route_sequence : UInt64
}

///|
/// Opaque identity for one framed submission. It carries neither a scheduler
/// ticket nor mutable request authority and can only be obtained from the
/// coordinator that admitted the frame.
#valtype
pub struct LunaOnlineFramedRequestRoute {
  priv sequence : UInt64
}

///|
#valtype
pub(all) enum LunaOnlineFramedCoordinatorProgress {
  LunaOnlineFramedCoordinatorIdle
  LunaOnlineFramedCoordinatorWaiting
  LunaOnlineFramedCoordinatorAwaitingInput
  LunaOnlineFramedCoordinatorAdvanced
  LunaOnlineFramedCoordinatorEventPreparing
  LunaOnlineFramedCoordinatorEventReady
  LunaOnlineFramedCoordinatorSemanticEventReady
  LunaOnlineFramedCoordinatorObservationReady
  LunaOnlineFramedCoordinatorRejectionReady
  LunaOnlineFramedCoordinatorMaintenanceRequired
  LunaOnlineFramedCoordinatorDraining
  LunaOnlineFramedCoordinatorClosed
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaOnlineFramedRejectionRule {
  LunaOnlineFramedRejectedFrame
  LunaOnlineFramedRejectedIdentity
  LunaOnlineFramedRejectedDeadline
  LunaOnlineFramedRejectedInput
  LunaOnlineFramedRejectedTokenization
  LunaOnlineFramedRejectedCapacity
  LunaOnlineFramedRejectedService
} derive(Debug, Eq)

///|
#valtype
pub struct LunaOnlineFramedRejectionCredit {
  priv service : LunaOnlineFramedService
  priv stream_epoch : UInt64
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaOnlineFramedEventOffer {
  priv service : LunaOnlineFramedService
  priv stream_epoch : UInt64
  priv event_epoch : UInt64
  priv offer_epoch : UInt64
  priv length : Int
}

///|
/// Exact one-event semantic delivery authority for a semantic-mode stream.
/// The service retains the lower event credit and owns its acknowledgement.
#valtype
pub struct LunaOnlineFramedSemanticEvent {
  priv service : LunaOnlineFramedService
  priv stream_epoch : UInt64
  priv event_epoch : UInt64
  priv route_sequence : UInt64
}

///|
#valtype
pub(all) enum LunaOnlineFramedObservationKind {
  LunaOnlineFramedAdmissionObservation
  LunaOnlineFramedTokenObservation
  LunaOnlineFramedUsageObservation
  LunaOnlineFramedCompletionObservation
  LunaOnlineFramedCancellationObservation
  LunaOnlineFramedDeadlineObservation
  LunaOnlineFramedRequestFailureObservation
  LunaOnlineFramedWorkerFailureObservation
  LunaOnlineFramedWorkerRestartObservation
  LunaOnlineFramedWorkerRequestTerminalObservation
} derive(Debug, Eq)

///|
/// Relative timing carried by one authenticated observation. The value is an
/// interval only; request identity and the monotonic clock origin stay private.
#valtype
pub(all) enum LunaOnlineFramedLatencyKind {
  LunaOnlineFramedNoLatency
  LunaOnlineFramedRequestLatency
  LunaOnlineFramedFirstTokenLatency
  LunaOnlineFramedInterTokenLatency
} derive(Debug, Eq)

///|
/// Exact one-pulse finite observability authority. No request identity,
/// payload, timestamp, or mutable telemetry owner is exposed.
#valtype
pub struct LunaOnlineFramedObservation {
  priv service : LunaOnlineFramedService
  priv stream_epoch : UInt64
  priv observation_epoch : UInt64
}

///|
#valtype
pub(all) enum LunaOnlineFramedServiceReadiness {
  LunaOnlineFramedServiceReady
  LunaOnlineFramedServiceStreamActive
  LunaOnlineFramedServiceMaintenanceRequired
  LunaOnlineFramedServiceDraining
  LunaOnlineFramedServiceClosed
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaOnlineFramedOpenKind {
  LunaOnlineFramedOpened
  LunaOnlineFramedOpenBusy
  LunaOnlineFramedOpenDraining
  LunaOnlineFramedOpenClosed
} derive(Debug, Eq)

///|
#valtype
pub struct LunaOnlineFramedOpen {
  priv disposition : Int
  priv epoch : UInt64
}

///|
/// Immutable scalar telemetry projection. It carries no scheduler, worker,
/// request, plan, page, stream epoch, or mutable owner authority.
pub struct LunaOnlineFramedTelemetry {
  priv queue_depth : Int
  priv active_requests : Int
  priv kv_pages_used : Int
  priv kv_pages_free : Int
  priv prefix_lookups : UInt64
  priv prefix_hits : UInt64
  priv prefix_misses : UInt64
  priv prefix_evictions : UInt64
  priv prefix_tokens_reused : UInt64
  priv prefix_tokens_computed : UInt64
  priv prefix_publications : UInt64
  priv prefix_entries : Int
  priv prefix_pages : Int
}

///|
#valtype
pub struct LunaOnlineFramedPlanTelemetry {
  priv plan_sequence : UInt64
  priv plan_rows : Int
  priv plan_tokens : Int
}

///|
#valtype
pub struct LunaOnlineFramedStream {
  priv service : LunaOnlineFramedService
  priv epoch : UInt64
}

///|
pub struct LunaOnlineFramedServicePreparation {
  priv mut service : LunaOnlineFramedService?
  priv mut cleanup : FailedLunaOnlineInstancePreparation?
  priv state : LunaOnlineInstancePreparationState
  priv mut consumed : Bool
}

///|
pub struct LunaOnlineFramedCoordinatorPreparation {
  priv mut coordinator : LunaOnlineFramedCoordinator?
  priv mut cleanup : FailedLunaOnlineInstancePreparation?
  priv state : LunaOnlineInstancePreparationState
  priv mut consumed : Bool
}

///|
pub struct LunaOnlineFramedService {
  priv pool : @request_admission.LunaRequestPreparationPool
  priv instance : LunaOnlineInstance
  priv outbound : @framed_wire.LunaFramedEventWorkspace
  priv event_copy_quantum : Int
  priv inference : @inference.InferenceLimits
  priv framed_limits : @framed_wire.FramedWireLimits
  priv expected_model : @spec.ModelIdentity
  priv clock : @monotonic_clock.MonotonicClock
  priv work_slots : FixedArray[
    Array[@request_admission.LunaRequestPreparationWork],
  ]
  priv queue_sequences : FixedArray[UInt64]
  priv queue_abandoned : FixedArray[Bool]
  priv prepared : Array[@request_admission.LunaPreparedRequest]
  priv tickets : Array[LunaOnlineRequestTicket]
  priv ticket_sequences : Array[UInt64]
  priv event_credits : Array[LunaOnlineEventCredit]
  priv outbound_works : Array[@framed_wire.LunaFramedEventWork]
  priv outbound_views : Array[@framed_wire.LunaFramedEventView]
  priv mut queue_head : Int
  priv mut queue_count : Int
  priv mut receipt_slot : Int
  priv mut next_sequence : UInt64
  priv mut event_kind : Int
  priv mut event_epoch : UInt64
  priv mut offer_epoch : UInt64
  priv mut event_length : Int
  priv mut confirmed_bytes : Int
  priv mut offered_bytes : Int
  priv mut semantic_event_pending : Bool
  priv mut event_ack_pending : Bool
  priv mut event_route_sequence : UInt64
  priv mut maintenance_kind : Int
  priv mut rejection_epoch : UInt64
  priv mut rejection_sequence : UInt64
  priv mut rejection_rule : Int
  priv mut rejection_taken : Bool
  priv mut stall_deadline_millis : UInt64
  priv mut stall_active : Bool
  priv mut turn_pool : Bool
  priv mut drain_requested : Bool
  priv mut disconnect_requested : Bool
  priv mut closed : Bool
  priv mut stream_epoch : UInt64
  priv mut stream_phase : Int
  priv mut stream_mode : Int
  priv stream_slots : Array[LunaOnlineFramedStream]
  priv mut semantic_event_taken : Bool
  priv mut observation_epoch : UInt64
  priv mut observation_kind : Int
  priv mut observation_taken : Bool
  priv mut observation_input_tokens : Int
  priv mut observation_cached_input_tokens : Int
  priv mut observation_output_tokens : Int
  priv mut observation_total_tokens : Int
  priv mut observation_latency_kind : Int
  priv mut observation_latency_millis : Int
  priv mut worker_failure_observed : Bool
  priv mut cooperative_handoff : Bool
  priv timing_slots : FixedArray[LunaOnlineFramedTimingSlot]
}

///|
/// Max-one compatibility facade over the reusable service and its first stream.
pub struct LunaOnlineFramedCoordinator {
  priv service : LunaOnlineFramedService
  priv stream : LunaOnlineFramedStream
}
