///|
fn LunaOnlineFramedService::begin_stall_deadline(
  self : LunaOnlineFramedService,
) -> Bool {
  if self.stall_active || self.stall_deadline_millis != 0UL {
    return false
  }
  let observed = self.clock.now_millis() catch { _ => return false }
  let budget = self.inference.max_deadline_millis()
  if budget == 0UL || observed > 0xffffffffffffffffUL - budget {
    return false
  }
  self.stall_deadline_millis = observed + budget
  self.stall_active = true
  true
}

///|
fn LunaOnlineFramedService::stall_expired(
  self : LunaOnlineFramedService,
) -> Int32 {
  if !self.stall_active {
    return 0
  }
  let observed = self.clock.now_millis() catch { _ => return 2 }
  if observed >= self.stall_deadline_millis {
    1
  } else {
    0
  }
}

///|
fn LunaOnlineFramedService::stall_remaining_millis(
  self : LunaOnlineFramedService,
) -> Int32 raise LunaOnlineFramedCoordinatorError {
  let maximum = self.inference.max_deadline_millis()
  if maximum == 0UL || maximum > 0x7fffffffUL {
    raise luna_coordinator_error(LunaOnlineFramedClock)
  }
  if !self.stall_active {
    if self.stall_deadline_millis != 0UL {
      raise luna_coordinator_error(LunaOnlineFramedLifecycle)
    }
    return maximum.to_int()
  }
  let observed = self.clock.now_millis() catch {
    _ => raise luna_coordinator_error(LunaOnlineFramedClock)
  }
  if observed >= self.stall_deadline_millis {
    return 0
  }
  let remaining = self.stall_deadline_millis - observed
  if remaining > maximum {
    raise luna_coordinator_error(LunaOnlineFramedClock)
  }
  remaining.to_int()
}

///|
/// Returns the shortest authenticated interval before the transport must poll
/// receipt or output-stall ownership again. It exposes no absolute timestamp.
fn LunaOnlineFramedService::stream_transport_wait_remaining_millis(
  self : LunaOnlineFramedService,
) -> Int32 raise LunaOnlineFramedCoordinatorError {
  if self.closed {
    return 0
  }
  let mut remaining = self.stall_remaining_millis()
  if self.receipt_slot >= 0 {
    let receipt = self
      .current_receipt_work()
      .luna_framed_receipt_remaining_millis() catch {
        ClockUnavailable => raise luna_coordinator_error(LunaOnlineFramedClock)
        _ => raise luna_coordinator_error(LunaOnlineFramedPreparation)
      }
    remaining = Int32::min(remaining, receipt)
  }
  remaining
}

///|
fn LunaOnlineFramedService::release_outbound_authority(
  self : LunaOnlineFramedService,
) -> Unit {
  if self.outbound_views.length() == 1 {
    try! self.outbound_views[0].release()
    self.outbound_views.clear()
  }
  if self.outbound_works.length() == 1 {
    try! self.outbound_works[0].abort()
    self.outbound_works.clear()
  }
  self.event_length = 0
  self.confirmed_bytes = 0
  self.offered_bytes = 0
}

///|
fn LunaOnlineFramedService::begin_output_disconnect(
  self : LunaOnlineFramedService,
) -> Unit raise LunaOnlineFramedCoordinatorError {
  self.disconnect_requested = true
  if self.stream_phase == 2 {
    self.stream_phase = 3
  }
  // Once terminal recovery, request retirement or cooperative multi-cut owns the cut, repeating
  // begin_abort is both
  // unnecessary and unsafe: the lower instance may already have advanced
  // beyond its abortable phase while maintenance still holds the ticket.
  if self.maintenance_kind == 1 ||
    self.maintenance_kind == 2 ||
    self.maintenance_kind == 4 {
    return
  }
  self.maintenance_kind = 0
  self.semantic_event_pending = false
  self.event_ack_pending = false
  self.event_route_sequence = 0UL
  self.semantic_event_taken = false
  self.stall_active = false
  self.stall_deadline_millis = 0UL
  self.release_outbound_authority()
  if self.tickets.length() == 0 {
    if self.event_credits.length() != 0 {
      raise luna_coordinator_error(LunaOnlineFramedMaintenance)
    }
    return
  }
  if self.ticket().lane < 0 {
    self.maintenance_kind = 2
    self.instance.begin_abort(self.ticket()) catch {
      _ => raise luna_coordinator_error(LunaOnlineFramedMaintenance)
    }
    self.event_credits.clear()
    return
  }
  self.maintenance_kind = 4
  self.event_credits.clear()
}

///|
fn LunaOnlineFramedService::enforce_output_stall(
  self : LunaOnlineFramedService,
) -> Int32 {
  let status = self.stall_expired()
  if status == 0 {
    return 0
  }
  self.begin_output_disconnect() catch {
    _ => ()
  }
  status
}
