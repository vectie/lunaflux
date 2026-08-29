///|
fn luna_framed_open(
  kind : LunaOnlineFramedOpenKind,
  epoch : UInt64,
) -> LunaOnlineFramedOpen {
  {
    disposition: match kind {
      LunaOnlineFramedOpened => 0
      LunaOnlineFramedOpenBusy => 1
      LunaOnlineFramedOpenDraining => 2
      LunaOnlineFramedOpenClosed => 3
    },
    epoch,
  }
}

///|
pub fn LunaOnlineFramedOpen::kind(
  self : LunaOnlineFramedOpen,
) -> LunaOnlineFramedOpenKind {
  match self.disposition {
    0 => LunaOnlineFramedOpened
    1 => LunaOnlineFramedOpenBusy
    2 => LunaOnlineFramedOpenDraining
    3 => LunaOnlineFramedOpenClosed
    _ => abort("Luna framed open invariant")
  }
}

///|
pub fn LunaOnlineFramedService::readiness(
  self : LunaOnlineFramedService,
) -> LunaOnlineFramedServiceReadiness {
  if self.closed {
    LunaOnlineFramedServiceClosed
  } else if self.drain_requested {
    LunaOnlineFramedServiceDraining
  } else if self.cooperative_handoff || self.maintenance_kind != 0 {
    LunaOnlineFramedServiceMaintenanceRequired
  } else if self.stream_phase != 0 {
    LunaOnlineFramedServiceStreamActive
  } else {
    LunaOnlineFramedServiceReady
  }
}

///|
/// Returns the next cooperative maintenance wake bound without progressing
/// cleanup or spawning a replacement.
pub fn LunaOnlineFramedService::maintenance_wait_remaining_millis(
  self : LunaOnlineFramedService,
) -> Int raise LunaOnlineFramedCoordinatorError {
  self.instance.maintenance_wait_remaining_millis() catch {
    _ => raise luna_coordinator_error(LunaOnlineFramedMaintenance)
  }
}

///|
fn LunaOnlineFramedService::open_stream_mode(
  self : LunaOnlineFramedService,
  mode : Int,
) -> LunaOnlineFramedOpen {
  if self.closed {
    return luna_framed_open(LunaOnlineFramedOpenClosed, 0UL)
  }
  if self.drain_requested {
    return luna_framed_open(LunaOnlineFramedOpenDraining, 0UL)
  }
  if self.stream_phase != 0 {
    return luna_framed_open(LunaOnlineFramedOpenBusy, 0UL)
  }
  if self.stream_slots.length() != 0 {
    abort("Luna framed stream slot invariant")
  }
  if self.stream_epoch == 0xffffffffffffffffUL {
    self.begin_service_drain()
    return luna_framed_open(LunaOnlineFramedOpenDraining, 0UL)
  }
  self.stream_epoch += 1UL
  self.stream_phase = 1
  self.stream_mode = mode
  self.stream_slots.push({ service: self, epoch: self.stream_epoch })
  luna_framed_open(LunaOnlineFramedOpened, self.stream_epoch)
}

///|
/// Opens one native event-v2 framed stream.
pub fn LunaOnlineFramedService::open_stream(
  self : LunaOnlineFramedService,
) -> LunaOnlineFramedOpen {
  self.open_stream_mode(1)
}

///|
/// Opens one semantic Luna-event stream. Semantic credits never expose the
/// framed adapter or its byte Offers.
pub fn LunaOnlineFramedService::open_semantic_stream(
  self : LunaOnlineFramedService,
) -> LunaOnlineFramedOpen {
  self.open_stream_mode(2)
}

///|
pub fn LunaOnlineFramedService::take_open_stream(
  self : LunaOnlineFramedService,
  opened : LunaOnlineFramedOpen,
) -> LunaOnlineFramedStream raise LunaOnlineFramedCoordinatorError {
  if opened.disposition != 0 ||
    opened.epoch == 0UL ||
    self.stream_phase != 1 ||
    self.stream_epoch != opened.epoch ||
    self.stream_slots.length() != 1 {
    raise luna_coordinator_error(LunaOnlineFramedOwnership)
  }
  self.stream_phase = 2
  let stream = self.stream_slots[0]
  self.stream_slots.clear()
  stream
}

///|
fn LunaOnlineFramedStream::require_service(
  self : LunaOnlineFramedStream,
) -> LunaOnlineFramedService raise LunaOnlineFramedCoordinatorError {
  let service = self.service
  if self.epoch == 0UL ||
    service.stream_epoch != self.epoch ||
    (service.stream_phase != 2 && service.stream_phase != 3) {
    raise luna_coordinator_error(LunaOnlineFramedOwnership)
  }
  service
}

///|
/// Reports whether this exact stream epoch still owns the service's active
/// stream. Retired and superseded aliases fail closed without exposing the
/// epoch or mutable service phase.
pub fn LunaOnlineFramedStream::is_live(self : LunaOnlineFramedStream) -> Bool {
  let service = self.service
  self.epoch != 0UL &&
  service.stream_epoch == self.epoch &&
  (service.stream_phase == 2 || service.stream_phase == 3)
}

///|
fn LunaOnlineFramedService::reset_retired_stream(
  self : LunaOnlineFramedService,
) -> Unit {
  if self.queue_count != 0 ||
    self.prepared.length() != 0 ||
    self.tickets.length() != 0 ||
    self.ticket_sequences.length() != 0 ||
    self.event_credits.length() != 0 ||
    self.outbound_works.length() != 0 ||
    self.outbound_views.length() != 0 ||
    self.observation_kind != 0 ||
    self.observation_taken ||
    self.pool.free_lane_count() != self.pool.lane_count() {
    abort("Luna framed stream retirement authority invariant")
  }
  self.receipt_slot = -1
  self.event_kind = 0
  self.event_length = 0
  self.confirmed_bytes = 0
  self.offered_bytes = 0
  self.semantic_event_pending = false
  self.event_ack_pending = false
  self.event_route_sequence = 0UL
  self.semantic_event_taken = false
  self.observation_kind = 0
  self.observation_taken = false
  self.observation_input_tokens = 0
  self.observation_cached_input_tokens = 0
  self.observation_output_tokens = 0
  self.observation_total_tokens = 0
  self.observation_latency_kind = 0
  self.observation_latency_millis = 0
  self.maintenance_kind = 0
  self.rejection_epoch = 0UL
  self.rejection_sequence = 0UL
  self.rejection_rule = 0
  self.rejection_taken = false
  self.stall_deadline_millis = 0UL
  self.stall_active = false
  self.turn_pool = true
  self.worker_failure_observed = false
  self.disconnect_requested = false
  self.cooperative_handoff = false
  self.clear_request_timing()
  self.stream_phase = 0
  self.stream_mode = 0
  if self.drain_requested {
    self.pool.begin_drain()
    self.instance.begin_drain()
  }
}

///|
fn LunaOnlineFramedService::stream_retirement_complete(
  self : LunaOnlineFramedService,
) -> Bool {
  self.disconnect_requested &&
  self.tickets.length() == 0 &&
  self.ticket_sequences.length() == 0 &&
  self.queue_count == 0 &&
  self.prepared.length() == 0 &&
  self.pool.free_lane_count() == self.pool.lane_count() &&
  self.event_credits.length() == 0 &&
  self.outbound_works.length() == 0 &&
  self.outbound_views.length() == 0 &&
  self.observation_kind == 0 &&
  !self.observation_taken
}

///|
fn LunaOnlineFramedService::finish_stream_retirement_if_ready(
  self : LunaOnlineFramedService,
) -> Bool {
  if !self.stream_retirement_complete() {
    return false
  }
  self.reset_retired_stream()
  true
}

///|
fn LunaOnlineFramedService::begin_service_drain(
  self : LunaOnlineFramedService,
) -> Unit {
  if self.closed || self.drain_requested {
    return
  }
  self.drain_requested = true
  if self.stream_phase == 1 {
    if self.stream_slots.length() != 1 {
      abort("Luna framed unopened stream slot invariant")
    }
    self.stream_slots.clear()
    self.stream_phase = 0
    self.stream_mode = 0
  }
  if self.stream_phase == 2 || self.stream_phase == 3 {
    self.begin_stream_disconnect() catch {
      _ => ()
    }
  } else {
    self.pool.begin_drain()
    self.instance.begin_drain()
  }
}

///|
pub fn LunaOnlineFramedService::begin_drain(
  self : LunaOnlineFramedService,
) -> Unit {
  self.begin_service_drain()
}

///|
pub fn LunaOnlineFramedService::is_closed(
  self : LunaOnlineFramedService,
) -> Bool {
  self.closed
}
