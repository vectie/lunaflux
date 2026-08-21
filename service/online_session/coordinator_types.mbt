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
}

///|
#valtype
pub(all) enum LunaOnlineFramedCoordinatorProgress {
  LunaOnlineFramedCoordinatorIdle
  LunaOnlineFramedCoordinatorAdvanced
  LunaOnlineFramedCoordinatorEventPreparing
  LunaOnlineFramedCoordinatorEventReady
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
  priv coordinator : LunaOnlineFramedCoordinator
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaOnlineFramedEventOffer {
  priv coordinator : LunaOnlineFramedCoordinator
  priv event_epoch : UInt64
  priv offer_epoch : UInt64
  priv length : Int
}

///|
pub struct LunaOnlineFramedCoordinatorPreparation {
  priv mut coordinator : LunaOnlineFramedCoordinator?
  priv mut cleanup : FailedLunaOnlineInstancePreparation?
  priv state : LunaOnlineInstancePreparationState
  priv mut consumed : Bool
}

///|
pub struct LunaOnlineFramedCoordinator {
  priv pool : @request_admission.LunaRequestPreparationPool
  priv instance : LunaOnlineInstance
  priv outbound : @framed_wire.LunaFramedEventWorkspace
  priv event_copy_quantum : Int
  priv inference : @inference.InferenceLimits
  priv clock : @monotonic_clock.MonotonicClock
  priv work_slots : FixedArray[
    Array[@request_admission.LunaRequestPreparationWork],
  ]
  priv queue_sequences : FixedArray[UInt64]
  priv prepared : Array[@request_admission.LunaPreparedRequest]
  priv tickets : Array[LunaOnlineRequestTicket]
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
}
