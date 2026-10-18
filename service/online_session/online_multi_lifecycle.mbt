///|
fn LunaOnlineInstance::reset_multi_lane(
  self : LunaOnlineInstance,
  lane_index : Int,
) -> Unit {
  let lane = self.multi.lanes[lane_index]
  if lane.claims.length() != 0 || lane.worker_requests.length() != 0 {
    abort("Luna online multi lane reset invariant")
  }
  lane.input_tokens = 0
  lane.cached_input_tokens = 0
  lane.accepted_acked = false
  lane.output_tokens = 0
  lane.final_tail_length = 0
  lane.finish_kind = FinishPending
  lane.string_cut = NoStringCut
  lane.terminal_cut = NoTerminalCut
  lane.deferred_cancel = false
  lane.failure_kind = NoLatchedFailure
  lane.restarted_failure_terminal_pending = false
  lane.recovery_close = false
  lane.phase = LunaInstanceIdlePhase
}

///|
fn LunaOnlineInstance::release_multi_lane(
  self : LunaOnlineInstance,
  lane_index : Int,
) -> Unit {
  let lane = self.multi.lanes[lane_index]
  if lane.claims.length() != 1 || lane.worker_requests.length() != 1 {
    abort("Luna online multi release invariant")
  }
  let request = lane.worker_requests[0]
  let route = try! request.publication_route()
  try! request.retire_terminal()
  self.multi.routes.clear(route, lane_index)
  try! lane.claims[0].release()
  lane.worker_requests.clear()
  lane.claims.clear()
  self.reset_multi_lane(lane_index)
  if self.multi.free_count < 0 ||
    self.multi.free_count >= self.multi.free_lanes.length() {
    abort("Luna online multi free ring release invariant")
  }
  let free_tail = (self.multi.free_head + self.multi.free_count) %
    self.multi.free_lanes.length()
  self.multi.free_lanes[free_tail] = lane_index
  self.multi.free_count += 1
  if self.multi.retirement_lane == lane_index {
    self.multi.retirement_lane = -1
  }
  self.multi.active_count -= 1
  if self.multi.active_count < 0 {
    abort("Luna online multi active count invariant")
  }
  if self.multi.active_count == 0 {
    self.phase = if self.drain_requested {
      LunaInstanceDrainingPhase
    } else {
      LunaInstanceIdlePhase
    }
  }
}

///|
fn LunaOnlineInstance::multi_retirement_ticket(
  self : LunaOnlineInstance,
) -> LunaOnlineRequestTicket {
  let lane = self.multi.retirement_lane
  if lane < 0 || lane >= self.multi.lanes.length() {
    abort("Luna online multi retirement lane invariant")
  }
  { lane, epoch: self.multi.lanes[lane].generation, }
}

///|
fn LunaOnlineInstance::request_cancel_multi(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
) -> Unit raise LunaOnlineInstanceError {
  let lane = self.require_active_multi_ticket(ticket)
  self.require_owner()
  if lane.terminal_cut != NoTerminalCut || lane.string_cut != NoStringCut {
    return
  }
  if self.events.has_event() {
    lane.deferred_cancel = true
    return
  }
  ignore(self.multi_begin_cancel(ticket.lane, CallerCancellationCut))
}

///|
fn LunaOnlineInstance::check_deadline_multi(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
) -> Unit raise LunaOnlineInstanceError {
  let lane = self.require_active_multi_ticket(ticket)
  self.require_owner()
  if lane.terminal_cut != NoTerminalCut ||
    lane.string_cut != NoStringCut ||
    lane.deferred_cancel ||
    self.events.has_event() {
    return
  }
  let expiry = self.multi_worker_request(ticket.lane).expire() catch {
      _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
    }
  if expiry is OnlineDeadlineCommitted {
    lane.failure_kind = DeadlineLatchedFailure
    lane.terminal_cut = DeadlineFailureCut
  }
}

///|
fn LunaOnlineInstance::finish_multi_abort_after_event_cut(
  self : LunaOnlineInstance,
  lane_index : Int,
) -> Bool {
  match self.event_kind {
    CompletionUsageEvent | CompletedEvent | FailureUsageEvent | FailedEvent => {
      self.multi.lanes[lane_index].phase = LunaRequestReleaseReadyPhase
      self.multi.retirement_lane = lane_index
      true
    }
    NoSessionEvent | AcceptedEvent | TokenEvent => false
  }
}

///|
fn LunaOnlineInstance::begin_abort_multi(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
) -> Unit raise LunaOnlineInstanceError {
  let lane = self.require_multi_ticket(ticket)
  self.require_owner()
  if lane.phase != LunaRequestActivePhase {
    return
  }
  if self.events.has_event() {
    if self.multi.event_lane != ticket.lane {
      lane.deferred_cancel = true
      return
    }
    try! self.events.discard()
    self.event_taken = false
    self.multi.event_lane = -1
    if self.finish_multi_abort_after_event_cut(ticket.lane) {
      return
    }
    match self.event_kind {
      AcceptedEvent => lane.accepted_acked = true
      NoSessionEvent
      | TokenEvent
      | CompletionUsageEvent
      | CompletedEvent
      | FailureUsageEvent
      | FailedEvent => ()
    }
  }
  if lane.string_cut == NoStringCut && lane.terminal_cut == NoTerminalCut {
    ignore(self.multi_begin_cancel(ticket.lane, CallerCancellationCut))
  }
  lane.phase = LunaRequestAwaitingTerminalPhase
}

///|
fn LunaOnlineInstance::multi_ticket_needs_abort(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
) -> Bool {
  if ticket.lane < 0 || ticket.lane >= self.multi.lanes.length() {
    return false
  }
  let lane = self.multi.lanes[ticket.lane]
  lane.generation == ticket.epoch && lane.phase == LunaRequestActivePhase
}

///|
fn LunaOnlineInstance::progress_multi_disconnect(
  self : LunaOnlineInstance,
) -> LunaOnlineInstanceProgress raise LunaOnlineInstanceError {
  if self.events.has_event() {
    return self.discard_multi_disconnect_event()
  }
  let progress = self.progress_multi()
  if self.events.has_event() {
    return self.discard_multi_disconnect_event()
  }
  progress
}

///|
/// A cut transport discards the held event on its actual route. Only an event
/// backed by a consumed terminal authorizes release; accepted/token events do
/// not. In particular the next publication need not belong to the ticket whose
/// cancellation was most recently progressed.
fn LunaOnlineInstance::discard_multi_disconnect_event(
  self : LunaOnlineInstance,
) -> LunaOnlineInstanceProgress raise LunaOnlineInstanceError {
  let lane_index = self.multi.event_lane
  if lane_index < 0 ||
    lane_index >= self.multi.lanes.length() ||
    !self.events.has_event() {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit))
  }
  try! self.events.discard()
  self.event_taken = false
  self.multi.event_lane = -1
  let terminal = self.finish_multi_abort_after_event_cut(lane_index)
  if self.event_kind is AcceptedEvent {
    self.multi.lanes[lane_index].accepted_acked = true
  }
  self.event_kind = NoSessionEvent
  if terminal {
    LunaOnlineInstanceRequestRetirementRequired
  } else {
    LunaOnlineInstancePending
  }
}

///|
fn LunaOnlineInstance::progress_multi_retirement(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
) -> LunaOnlineInstanceRetirementProgress raise LunaOnlineInstanceError {
  let lane = self.require_multi_ticket(ticket)
  self.require_owner()
  if self.recovery_close {
    ignore(self.advance_multi_recovery_maintenance())
    return LunaOnlineInstanceRetirementPending
  }
  if lane.phase == LunaRequestAwaitingTerminalPhase {
    if self.events.has_event() {
      return LunaOnlineInstanceRetirementPending
    }
    ignore(self.multi_route_one_publication())
    if self.events.has_event() {
      ignore(self.discard_multi_disconnect_event())
    }
    return LunaOnlineInstanceRetirementPending
  }
  if lane.phase != LunaRequestReleaseReadyPhase {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineLifecycle))
  }
  if !(self.multi_worker_request(ticket.lane).progress_terminal_retirement() catch {
      _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
    }) {
    return LunaOnlineInstanceRetirementPending
  }
  self.release_multi_lane(ticket.lane)
  LunaOnlineInstanceIdle
}
