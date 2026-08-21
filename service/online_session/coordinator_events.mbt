///|
fn LunaOnlineFramedCoordinator::ticket(
  self : LunaOnlineFramedCoordinator,
) -> LunaOnlineRequestTicket {
  if self.tickets.length() != 1 {
    abort("Luna coordinator ticket invariant")
  }
  self.tickets[0]
}

///|
fn luna_online_event_kind(kind : @luna_event.LunaEventKind) -> Int {
  match kind {
    LunaAcceptedEventKind => 1
    LunaTokenEventKind => 2
    LunaUsageEventKind => 3
    LunaCompletedEventKind => 4
    LunaFailedEventKind => 5
  }
}

///|
fn LunaOnlineFramedCoordinator::begin_stall_deadline(
  self : LunaOnlineFramedCoordinator,
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
fn LunaOnlineFramedCoordinator::stall_expired(
  self : LunaOnlineFramedCoordinator,
) -> Int {
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
fn LunaOnlineFramedCoordinator::release_outbound_authority(
  self : LunaOnlineFramedCoordinator,
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
fn LunaOnlineFramedCoordinator::begin_output_disconnect(
  self : LunaOnlineFramedCoordinator,
) -> Unit raise LunaOnlineFramedCoordinatorError {
  self.disconnect_requested = true
  self.drain_requested = true
  self.maintenance_kind = 0
  self.semantic_event_pending = false
  self.event_ack_pending = false
  self.stall_active = false
  self.stall_deadline_millis = 0UL
  self.release_outbound_authority()
  self.pool.begin_drain()
  self.instance.begin_drain()
  if self.tickets.length() != 1 {
    if self.event_credits.length() != 0 {
      raise luna_coordinator_error(LunaOnlineFramedMaintenance)
    }
    return
  }
  self.maintenance_kind = 2
  self.instance.begin_abort(self.ticket()) catch {
    _ => raise luna_coordinator_error(LunaOnlineFramedMaintenance)
  }
  self.event_credits.clear()
}

///|
fn LunaOnlineFramedCoordinator::enforce_output_stall(
  self : LunaOnlineFramedCoordinator,
) -> Int {
  let status = self.stall_expired()
  if status == 0 {
    return 0
  }
  self.begin_output_disconnect() catch {
    _ => ()
  }
  status
}

///|
fn LunaOnlineFramedCoordinator::start_semantic_event(
  self : LunaOnlineFramedCoordinator,
) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError {
  if self.tickets.length() != 1 || self.event_credits.length() != 0 {
    raise luna_coordinator_error(LunaOnlineFramedLifecycle)
  }
  if self.event_epoch == 0xffffffffffffffffUL {
    self.begin_output_disconnect() catch {
      _ => ()
    }
    return LunaOnlineFramedCoordinatorMaintenanceRequired
  }
  let credit = self.instance.take_event(self.ticket()) catch {
    _ => raise luna_coordinator_error(LunaOnlineFramedEvent)
  }
  self.event_credits.push(credit)
  self.event_epoch += 1UL
  let semantic = credit.view() catch {
    _ => {
      self.begin_output_disconnect() catch {
        _ => ()
      }
      raise luna_coordinator_error(LunaOnlineFramedEvent)
    }
  }
  self.event_kind = luna_online_event_kind(
    semantic.kind() catch {
      _ => {
        self.begin_output_disconnect() catch {
          _ => ()
        }
        raise luna_coordinator_error(LunaOnlineFramedEvent)
      }
    },
  )
  if !self.begin_stall_deadline() {
    self.begin_output_disconnect() catch {
      _ => ()
    }
    return LunaOnlineFramedCoordinatorMaintenanceRequired
  }
  let work = self.outbound.begin(semantic) catch {
    _ => {
      self.begin_output_disconnect() catch {
        _ => ()
      }
      raise luna_coordinator_error(LunaOnlineFramedEvent)
    }
  }
  self.outbound_works.push(work)
  self.semantic_event_pending = false
  LunaOnlineFramedCoordinatorEventPreparing
}

///|
fn LunaOnlineFramedCoordinator::finish_event_ack(
  self : LunaOnlineFramedCoordinator,
) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError {
  if self.event_credits.length() != 1 || !self.event_ack_pending {
    raise luna_coordinator_error(LunaOnlineFramedLifecycle)
  }
  self.event_credits[0].ack() catch {
    _ => raise luna_coordinator_error(LunaOnlineFramedEvent)
  }
  self.event_credits.clear()
  self.event_ack_pending = false
  self.stall_active = false
  self.stall_deadline_millis = 0UL
  let kind = self.event_kind
  self.event_kind = 0
  if kind == 3 {
    self.maintenance_kind = 0
    self.semantic_event_pending = true
    return LunaOnlineFramedCoordinatorAdvanced
  }
  if kind == 4 || kind == 5 {
    self.maintenance_kind = 2
    return LunaOnlineFramedCoordinatorMaintenanceRequired
  }
  LunaOnlineFramedCoordinatorAdvanced
}

///|
fn LunaOnlineFramedCoordinator::progress_outbound_event(
  self : LunaOnlineFramedCoordinator,
) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError {
  if self.enforce_output_stall() != 0 {
    return LunaOnlineFramedCoordinatorMaintenanceRequired
  }
  if self.event_ack_pending {
    if self.event_kind == 3 {
      self.maintenance_kind = 3
      return LunaOnlineFramedCoordinatorMaintenanceRequired
    }
    return self.finish_event_ack()
  }
  if self.outbound_views.length() == 1 {
    return LunaOnlineFramedCoordinatorEventReady
  }
  if self.outbound_works.length() != 1 {
    raise luna_coordinator_error(LunaOnlineFramedLifecycle)
  }
  let work = self.outbound_works[0]
  let state = work.state() catch {
    _ => raise luna_coordinator_error(LunaOnlineFramedEvent)
  }
  match state {
    LunaFramedEventFailed => {
      ignore(work.failure() catch { _ => LunaFramedEventFailed })
      try! work.abort()
      self.outbound_works.clear()
      self.begin_output_disconnect() catch {
        _ => ()
      }
      LunaOnlineFramedCoordinatorMaintenanceRequired
    }
    LunaFramedEventReady => {
      let view = work.take_view() catch {
        _ => raise luna_coordinator_error(LunaOnlineFramedEvent)
      }
      self.outbound_works.clear()
      self.outbound_views.push(view)
      self.event_length = view.length() catch {
        _ => raise luna_coordinator_error(LunaOnlineFramedEvent)
      }
      self.confirmed_bytes = 0
      self.offered_bytes = 0
      LunaOnlineFramedCoordinatorEventReady
    }
    LunaFramedEventPending => {
      ignore(
        work.progress() catch {
          _ => raise luna_coordinator_error(LunaOnlineFramedEvent)
        },
      )
      LunaOnlineFramedCoordinatorEventPreparing
    }
  }
}

///|
pub fn LunaOnlineFramedCoordinator::framed_event_remaining(
  self : LunaOnlineFramedCoordinator,
) -> Int raise LunaOnlineFramedCoordinatorError {
  if self.enforce_output_stall() != 0 {
    raise luna_coordinator_error(LunaOnlineFramedOutputStall)
  }
  if self.outbound_views.length() != 1 {
    raise luna_coordinator_error(LunaOnlineFramedEvent)
  }
  self.event_length - self.confirmed_bytes
}

///|
pub fn LunaOnlineFramedCoordinator::copy_framed_event_chunk_to(
  self : LunaOnlineFramedCoordinator,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
  maximum_length~ : Int,
) -> LunaOnlineFramedEventOffer raise LunaOnlineFramedCoordinatorError {
  if self.enforce_output_stall() != 0 {
    raise luna_coordinator_error(LunaOnlineFramedOutputStall)
  }
  if self.outbound_views.length() != 1 || self.offered_bytes != 0 {
    raise luna_coordinator_error(LunaOnlineFramedEvent)
  }
  let remaining = self.event_length - self.confirmed_bytes
  if maximum_length <= 0 || remaining <= 0 {
    raise luna_coordinator_error(LunaOnlineFramedEvent)
  }
  if self.offer_epoch == 0xffffffffffffffffUL {
    self.begin_output_disconnect() catch {
      _ => ()
    }
    raise luna_coordinator_error(LunaOnlineFramedOwnership)
  }
  let amount = Int::min(
    Int::min(maximum_length, self.event_copy_quantum),
    remaining,
  )
  ignore(
    self.outbound_views[0].copy_chunk_to(
      destination,
      destination_offset~,
      source_offset=self.confirmed_bytes,
      length=amount,
    ) catch {
      _ => raise luna_coordinator_error(LunaOnlineFramedEvent)
    },
  )
  self.offered_bytes = amount
  self.offer_epoch += 1UL
  {
    coordinator: self,
    event_epoch: self.event_epoch,
    offer_epoch: self.offer_epoch,
    length: amount,
  }
}

///|
fn LunaOnlineFramedEventOffer::require_owner(
  self : LunaOnlineFramedEventOffer,
) -> LunaOnlineFramedCoordinator raise LunaOnlineFramedCoordinatorError {
  let owner = self.coordinator
  if self.event_epoch == 0UL ||
    self.offer_epoch == 0UL ||
    owner.event_epoch != self.event_epoch ||
    owner.offer_epoch != self.offer_epoch ||
    owner.offered_bytes != self.length ||
    owner.outbound_views.length() != 1 {
    raise luna_coordinator_error(LunaOnlineFramedEvent)
  }
  owner
}

///|
pub fn LunaOnlineFramedEventOffer::length(
  self : LunaOnlineFramedEventOffer,
) -> Int raise LunaOnlineFramedCoordinatorError {
  ignore(self.require_owner())
  self.length
}

///|
pub fn LunaOnlineFramedEventOffer::confirm(
  self : LunaOnlineFramedEventOffer,
  length~ : Int,
) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError {
  let owner = self.require_owner()
  if owner.enforce_output_stall() != 0 {
    raise luna_coordinator_error(LunaOnlineFramedOutputStall)
  }
  if length <= 0 || length > self.length {
    raise luna_coordinator_error(LunaOnlineFramedEvent)
  }
  let next_confirmed = owner.confirmed_bytes + length
  if next_confirmed < owner.event_length {
    owner.confirmed_bytes = next_confirmed
    owner.offered_bytes = 0
    return LunaOnlineFramedCoordinatorEventReady
  }
  // Exact authentication above proves this is the sole live framed View.
  // Release before invalidating the Offer or mutating confirmed cursors.
  try! owner.outbound_views[0].release()
  owner.outbound_views.clear()
  owner.offered_bytes = 0
  owner.event_length = 0
  owner.confirmed_bytes = 0
  owner.event_ack_pending = true
  if owner.event_kind == 3 {
    owner.maintenance_kind = 3
    return LunaOnlineFramedCoordinatorMaintenanceRequired
  }
  owner.finish_event_ack()
}
