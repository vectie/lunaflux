///|
fn LunaOnlineFramedService::ticket(
  self : LunaOnlineFramedService,
) -> LunaOnlineRequestTicket {
  if self.tickets.length() == 0 {
    abort("Luna coordinator ticket invariant")
  }
  self.tickets[0]
}

///|
fn LunaOnlineFramedService::route_sequence_for_ticket(
  self : LunaOnlineFramedService,
  ticket : LunaOnlineRequestTicket,
) -> UInt64 {
  if self.ticket_sequences.length() != self.tickets.length() {
    abort("Luna coordinator ticket route invariant")
  }
  for index, current in self.tickets {
    if current.lane == ticket.lane && current.epoch == ticket.epoch {
      let sequence = self.ticket_sequences[index]
      if sequence == 0UL {
        abort("Luna coordinator ticket route invariant")
      }
      return sequence
    }
  }
  abort("Luna coordinator event route invariant")
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
fn LunaOnlineFramedService::stall_remaining_millis(
  self : LunaOnlineFramedService,
) -> Int raise LunaOnlineFramedCoordinatorError {
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
) -> Int raise LunaOnlineFramedCoordinatorError {
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
    remaining = Int::min(remaining, receipt)
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
  // Once request retirement or cooperative multi-cut owns the cut, repeating
  // begin_abort is both
  // unnecessary and unsafe: the lower instance may already have advanced
  // beyond its abortable phase while maintenance still holds the ticket.
  if self.maintenance_kind == 2 || self.maintenance_kind == 4 {
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
fn LunaOnlineFramedService::start_semantic_event(
  self : LunaOnlineFramedService,
) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError {
  if self.tickets.length() == 0 || self.event_credits.length() != 0 {
    raise luna_coordinator_error(LunaOnlineFramedLifecycle)
  }
  if self.event_epoch == 0xffffffffffffffffUL {
    self.begin_output_disconnect() catch {
      _ => ()
    }
    return LunaOnlineFramedCoordinatorMaintenanceRequired
  }
  let event_ticket = self.instance.multi_event_ticket()
  let credit = self.instance.take_event(event_ticket) catch {
    _ => raise luna_coordinator_error(LunaOnlineFramedEvent)
  }
  self.event_route_sequence = self.route_sequence_for_ticket(event_ticket)
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
  self.semantic_event_pending = false
  self.semantic_event_taken = false
  let observation_ready = self.publish_event_observation(semantic)
  if self.stream_mode == 2 {
    return if observation_ready {
      LunaOnlineFramedCoordinatorObservationReady
    } else {
      LunaOnlineFramedCoordinatorSemanticEventReady
    }
  }
  if self.stream_mode != 1 {
    self.begin_output_disconnect() catch {
      _ => ()
    }
    raise luna_coordinator_error(LunaOnlineFramedLifecycle)
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
  if observation_ready {
    LunaOnlineFramedCoordinatorObservationReady
  } else {
    LunaOnlineFramedCoordinatorEventPreparing
  }
}

///|
fn LunaOnlineFramedService::finish_event_ack(
  self : LunaOnlineFramedService,
) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError {
  if self.event_credits.length() != 1 || !self.event_ack_pending {
    raise luna_coordinator_error(LunaOnlineFramedLifecycle)
  }
  self.event_credits[0].ack() catch {
    _ => raise luna_coordinator_error(LunaOnlineFramedEvent)
  }
  self.event_credits.clear()
  self.event_ack_pending = false
  self.semantic_event_taken = false
  self.stall_active = false
  self.stall_deadline_millis = 0UL
  let kind = self.event_kind
  self.event_kind = 0
  self.event_route_sequence = 0UL
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
fn LunaOnlineFramedService::progress_outbound_event(
  self : LunaOnlineFramedService,
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
  if self.stream_mode == 2 {
    if self.event_credits.length() != 1 ||
      self.outbound_views.length() != 0 ||
      self.outbound_works.length() != 0 {
      raise luna_coordinator_error(LunaOnlineFramedLifecycle)
    }
    return LunaOnlineFramedCoordinatorSemanticEventReady
  }
  if self.stream_mode != 1 {
    raise luna_coordinator_error(LunaOnlineFramedLifecycle)
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
fn LunaOnlineFramedService::stream_framed_event_remaining(
  self : LunaOnlineFramedService,
) -> Int raise LunaOnlineFramedCoordinatorError {
  if self.stream_mode != 1 {
    raise luna_coordinator_error(LunaOnlineFramedOwnership)
  }
  if self.enforce_output_stall() != 0 {
    raise luna_coordinator_error(LunaOnlineFramedOutputStall)
  }
  if self.outbound_views.length() != 1 {
    raise luna_coordinator_error(LunaOnlineFramedEvent)
  }
  self.event_length - self.confirmed_bytes
}

///|
fn LunaOnlineFramedService::stream_copy_framed_event_chunk_to(
  self : LunaOnlineFramedService,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
  maximum_length~ : Int,
) -> LunaOnlineFramedEventOffer raise LunaOnlineFramedCoordinatorError {
  if self.stream_mode != 1 {
    raise luna_coordinator_error(LunaOnlineFramedOwnership)
  }
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
    service: self,
    stream_epoch: self.stream_epoch,
    event_epoch: self.event_epoch,
    offer_epoch: self.offer_epoch,
    length: amount,
  }
}

///|
fn LunaOnlineFramedEventOffer::require_owner(
  self : LunaOnlineFramedEventOffer,
) -> LunaOnlineFramedService raise LunaOnlineFramedCoordinatorError {
  let owner = self.service
  if self.event_epoch == 0UL ||
    self.stream_epoch == 0UL ||
    owner.stream_epoch != self.stream_epoch ||
    (owner.stream_phase != 2 && owner.stream_phase != 3) ||
    owner.stream_mode != 1 ||
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
