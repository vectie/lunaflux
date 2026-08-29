///|
#warnings("-unused_field")
priv struct LunaOnlineMultiLane {
  claims : Array[@request_admission.LunaPreparedRequestClaim]
  worker_requests : Array[@worker_service.LunaOnlineWorkerRequest]
  output : FixedArray[Byte]
  mut generation : UInt64
  mut input_tokens : Int
  mut cached_input_tokens : Int
  mut accepted_acked : Bool
  mut output_tokens : Int
  mut final_tail_length : Int
  mut finish_kind : SessionFinishKind
  mut string_cut : SessionStringCut
  mut terminal_cut : SessionTerminalCut
  mut deferred_cancel : Bool
  mut failure_kind : SessionLatchedFailure
  mut phase : LunaInstancePhase
  mut restarted_failure_terminal_pending : Bool
  mut recovery_close : Bool
}

///|
#warnings("-unused_field")
priv struct LunaOnlineMultiState {
  lanes : FixedArray[LunaOnlineMultiLane]
  free_lanes : FixedArray[Int]
  routes : LunaOnlinePublicationRoutes
  mut active_count : Int
  mut free_head : Int
  mut free_count : Int
  mut event_lane : Int
  mut progress_lane : Int
  mut retirement_lane : Int
  mut failure_mark_cursor : Int
  mut failure_mark_active : Bool
}

///|
fn new_luna_online_multi_lane(
  inference : @inference.InferenceLimits,
) -> LunaOnlineMultiLane {
  {
    claims: Array::new(capacity=1),
    worker_requests: Array::new(capacity=1),
    output: FixedArray::make(inference.max_decoded_delta_bytes(), b'\x00'),
    generation: 0UL,
    input_tokens: 0,
    cached_input_tokens: 0,
    accepted_acked: false,
    output_tokens: 0,
    final_tail_length: 0,
    finish_kind: FinishPending,
    string_cut: NoStringCut,
    terminal_cut: NoTerminalCut,
    deferred_cancel: false,
    failure_kind: NoLatchedFailure,
    phase: LunaInstanceIdlePhase,
    restarted_failure_terminal_pending: false,
    recovery_close: false,
  }
}

///|
fn new_luna_online_multi_state(
  capacity : Int,
  inference : @inference.InferenceLimits,
) -> LunaOnlineMultiState {
  if capacity <= 0 {
    abort("Luna online multi capacity invariant")
  }
  {
    lanes: FixedArray::makei(capacity, _ => {
      new_luna_online_multi_lane(inference)
    }),
    free_lanes: FixedArray::makei(capacity, index => index),
    routes: LunaOnlinePublicationRoutes::new(capacity),
    active_count: 0,
    free_head: 0,
    free_count: capacity,
    event_lane: -1,
    progress_lane: 0,
    retirement_lane: -1,
    failure_mark_cursor: 0,
    failure_mark_active: false,
  }
}
