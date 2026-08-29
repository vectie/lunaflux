///|
fn luna_coordinator_preparation_failure(
  rule : LunaOnlineFramedCoordinatorRule,
) -> LunaOnlineInstancePreparationError {
  let failure = match rule {
    LunaOnlineFramedOwnership => LunaOnlineInvalid(LunaOnlineOwnership)
    _ => LunaOnlineServicePreparationFailure
  }
  LunaOnlineInstancePreparationFailed(failure)
}

///|
fn new_luna_online_framed_coordinator(
  pool : @request_admission.LunaRequestPreparationPool,
  instance : LunaOnlineInstance,
  outbound : @framed_wire.LunaFramedEventWorkspace,
  event_copy_quantum : Int,
  inference : @inference.InferenceLimits,
  framed_limits : @framed_wire.FramedWireLimits,
  expected_model : @spec.ModelIdentity,
  clock : @monotonic_clock.MonotonicClock,
) -> LunaOnlineFramedService {
  let lane_count = pool.lane_count()
  {
    pool,
    instance,
    outbound,
    event_copy_quantum,
    inference,
    framed_limits,
    expected_model,
    clock,
    work_slots: FixedArray::makei(lane_count, _ => Array::new(capacity=1)),
    queue_sequences: FixedArray::make(lane_count, 0UL),
    queue_abandoned: FixedArray::make(lane_count, false),
    prepared: Array::new(capacity=1),
    tickets: Array::new(capacity=lane_count),
    ticket_sequences: Array::new(capacity=lane_count),
    event_credits: Array::new(capacity=1),
    outbound_works: Array::new(capacity=1),
    outbound_views: Array::new(capacity=1),
    queue_head: 0,
    queue_count: 0,
    receipt_slot: -1,
    next_sequence: 0UL,
    event_kind: 0,
    event_epoch: 0UL,
    offer_epoch: 0UL,
    event_length: 0,
    confirmed_bytes: 0,
    offered_bytes: 0,
    semantic_event_pending: false,
    event_ack_pending: false,
    event_route_sequence: 0UL,
    maintenance_kind: 0,
    rejection_epoch: 0UL,
    rejection_sequence: 0UL,
    rejection_rule: 0,
    rejection_taken: false,
    stall_deadline_millis: 0UL,
    stall_active: false,
    turn_pool: true,
    drain_requested: false,
    disconnect_requested: false,
    closed: false,
    stream_epoch: 0UL,
    stream_phase: 0,
    stream_mode: 0,
    stream_slots: Array::new(capacity=1),
    semantic_event_taken: false,
    observation_epoch: 0UL,
    observation_kind: 0,
    observation_taken: false,
    observation_input_tokens: 0,
    observation_cached_input_tokens: 0,
    observation_output_tokens: 0,
    observation_total_tokens: 0,
    observation_latency_kind: 0,
    observation_latency_millis: 0,
    worker_failure_observed: false,
    cooperative_handoff: false,
    timing_slots: new_luna_online_framed_timing_slots(
      instance.multi.lanes.length(),
    ),
  }
}
