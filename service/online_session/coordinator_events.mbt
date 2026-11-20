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
fn luna_online_event_kind(kind : @luna_event.LunaEventKind) -> Int32 {
  match kind {
    LunaAcceptedEventKind => 1
    LunaTokenEventKind => 2
    LunaUsageEventKind => 3
    LunaCompletedEventKind => 4
    LunaFailedEventKind => 5
  }
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
    if self.semantic_event_taken {
      return LunaOnlineFramedCoordinatorEventPreparing
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
