///|
fn LunaOnlineFramedService::request_cancel(
  self : LunaOnlineFramedService,
) -> Unit raise LunaOnlineFramedCoordinatorError {
  if self.tickets.length() == 0 || self.disconnect_requested {
    raise luna_coordinator_error(LunaOnlineFramedLifecycle)
  }
  self.instance.request_cancel(self.ticket()) catch {
    _ => raise luna_coordinator_error(LunaOnlineFramedMaintenance)
  }
}

///|
fn LunaOnlineFramedService::request_cancel_route(
  self : LunaOnlineFramedService,
  route : LunaOnlineFramedRequestRoute,
) -> Unit raise LunaOnlineFramedCoordinatorError {
  if route.sequence == 0UL || self.disconnect_requested {
    raise luna_coordinator_error(LunaOnlineFramedLifecycle)
  }
  if self.ticket_sequences.length() != self.tickets.length() {
    abort("Luna coordinator ticket route invariant")
  }
  for index, sequence in self.ticket_sequences {
    if sequence == route.sequence {
      self.instance.request_cancel(self.tickets[index]) catch {
        _ => raise luna_coordinator_error(LunaOnlineFramedMaintenance)
      }
      return
    }
  }
  for offset in 0..<self.queue_count {
    let slot = (self.queue_head + offset) % self.work_slots.length()
    if self.queue_sequences[slot] == route.sequence {
      self.queue_abandoned[slot] = true
      if slot == self.queue_head && self.prepared.length() == 1 {
        self.prepared[0].discard() catch {
          _ => raise luna_coordinator_error(LunaOnlineFramedPreparation)
        }
        self.prepared.clear()
        if self.receipt_slot == slot {
          self.receipt_slot = -1
        }
        self.pop_queue_head()
      } else {
        self.work_slots[slot][0].cancel() catch {
          _ => raise luna_coordinator_error(LunaOnlineFramedPreparation)
        }
      }
      return
    }
  }
  raise luna_coordinator_error(LunaOnlineFramedOwnership)
}

///|
fn LunaOnlineFramedService::request_route_live(
  self : LunaOnlineFramedService,
  route : LunaOnlineFramedRequestRoute,
) -> Bool {
  if route.sequence == 0UL ||
    self.ticket_sequences.length() != self.tickets.length() {
    return false
  }
  for sequence in self.ticket_sequences {
    if sequence == route.sequence {
      return true
    }
  }
  for offset in 0..<self.queue_count {
    let slot = (self.queue_head + offset) % self.work_slots.length()
    if self.queue_sequences[slot] == route.sequence {
      return true
    }
  }
  false
}

///|
/// Cuts the single transport-neutral stream. Framed authority is revoked
/// before semantic event authority, and all later cleanup remains owned by the
/// off-reactor maintenance path.
fn LunaOnlineFramedService::begin_stream_disconnect(
  self : LunaOnlineFramedService,
) -> Unit raise LunaOnlineFramedCoordinatorError {
  if self.closed {
    return
  }
  if self.stream_phase != 2 && self.stream_phase != 3 {
    raise luna_coordinator_error(LunaOnlineFramedOwnership)
  }
  self.stream_phase = 3
  self.disconnect_requested = true
  self.semantic_event_pending = false
  if self.rejection_epoch != 0UL {
    self.rejection_epoch = 0UL
    self.rejection_sequence = 0UL
    self.rejection_rule = 0
    self.rejection_taken = false
    self.stall_active = false
    self.stall_deadline_millis = 0UL
  }
  if !self.tickets.is_empty() {
    self.begin_output_disconnect()
  } else {
    self.release_outbound_authority()
    if self.event_credits.length() != 0 {
      raise luna_coordinator_error(LunaOnlineFramedMaintenance)
    }
  }
}

///|
fn LunaOnlineFramedService::finish_ticket_retirement(
  self : LunaOnlineFramedService,
  ticket : LunaOnlineRequestTicket,
  progress : LunaOnlineInstanceRetirementProgress,
) -> LunaOnlineFramedCoordinatorProgress {
  match progress {
    LunaOnlineInstanceRetirementPending =>
      LunaOnlineFramedCoordinatorMaintenanceRequired
    LunaOnlineInstanceIdle => {
      let mut found = -1
      for index, current in self.tickets {
        if current.lane == ticket.lane && current.epoch == ticket.epoch {
          found = index
          break
        }
      }
      if found < 0 {
        abort("Luna coordinator retired ticket invariant")
      }
      self.retire_request_timing(ticket)
      ignore(self.tickets.remove(found))
      ignore(self.ticket_sequences.remove(found))
      self.maintenance_kind = if self.disconnect_requested &&
        !self.tickets.is_empty() {
        4
      } else {
        0
      }
      if self.drain_requested {
        LunaOnlineFramedCoordinatorDraining
      } else {
        LunaOnlineFramedCoordinatorAdvanced
      }
    }
    LunaOnlineInstanceClosed => {
      self.clear_request_timing()
      self.tickets.clear()
      self.ticket_sequences.clear()
      self.maintenance_kind = 0
      // The lower owner is closed, but FIFO preparation lanes are independent
      // authorities. Cut admission and let reactor progress release every
      // queued Work/Prepared before the coordinator publishes Closed.
      self.drain_requested = true
      self.disconnect_requested = true
      self.pool.begin_drain()
      LunaOnlineFramedCoordinatorDraining
    }
  }
}

///|
/// Drives only transitions that may recover, reap, retire, or close lower
/// owners. Network-reactor code must never call this method.
fn LunaOnlineFramedService::progress_maintenance_core(
  self : LunaOnlineFramedService,
) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError {
  if self.closed {
    return LunaOnlineFramedCoordinatorClosed
  }
  if self.observation_kind != 0 {
    return LunaOnlineFramedCoordinatorObservationReady
  }
  if self.rejection_epoch != 0UL {
    if !self.enforce_rejection_stall() {
      return LunaOnlineFramedCoordinatorRejectionReady
    }
  }
  if self.disconnect_requested && !self.tickets.is_empty() {
    self.begin_output_disconnect() catch {
      _ => return LunaOnlineFramedCoordinatorMaintenanceRequired
    }
  }
  if self.stream_mode == 2 &&
    self.event_credits.length() == 1 &&
    !self.event_ack_pending &&
    !self.semantic_event_taken {
    if self.enforce_output_stall() != 0 {
      return LunaOnlineFramedCoordinatorMaintenanceRequired
    }
    return LunaOnlineFramedCoordinatorSemanticEventReady
  }
  if self.outbound_views.length() == 1 {
    return LunaOnlineFramedCoordinatorEventReady
  }
  if self.maintenance_kind == 3 {
    if self.enforce_output_stall() != 0 {
      return LunaOnlineFramedCoordinatorMaintenanceRequired
    }
    return self.finish_event_ack()
  }
  if self.event_ack_pending ||
    self.outbound_works.length() == 1 ||
    self.semantic_event_taken ||
    self.semantic_event_pending {
    return LunaOnlineFramedCoordinatorEventPreparing
  }
  if self.maintenance_kind == 1 {
    if self.tickets.length() == 0 {
      raise luna_coordinator_error(LunaOnlineFramedLifecycle)
    }
    let progress = self.instance.progress_terminalization(self.ticket()) catch {
      _ => raise luna_coordinator_error(LunaOnlineFramedMaintenance)
    }
    if self.publish_worker_failure_if_needed() {
      return LunaOnlineFramedCoordinatorObservationReady
    }
    if self.instance.take_restart_observation() {
      self.publish_observation(
        LunaOnlineFramedWorkerRestartObservation,
        0,
        0,
        0,
        0,
      )
      return LunaOnlineFramedCoordinatorObservationReady
    }
    match progress {
      LunaOnlineInstanceEventReady => {
        self.maintenance_kind = 0
        self.semantic_event_pending = true
        return LunaOnlineFramedCoordinatorEventPreparing
      }
      LunaOnlineInstanceTerminalizationRequired =>
        return LunaOnlineFramedCoordinatorMaintenanceRequired
      LunaOnlineInstanceRequestRetirementRequired => {
        self.maintenance_kind = 2
        return LunaOnlineFramedCoordinatorMaintenanceRequired
      }
      LunaOnlineInstancePending | LunaOnlineInstanceWaiting =>
        return LunaOnlineFramedCoordinatorMaintenanceRequired
    }
  }
  if self.maintenance_kind == 2 {
    if self.tickets.length() == 0 {
      raise luna_coordinator_error(LunaOnlineFramedLifecycle)
    }
    let ticket = if self.instance.multi.retirement_lane >= 0 {
      self.instance.multi_retirement_ticket()
    } else {
      self.ticket()
    }
    let retired = self.instance.progress_request_retirement(ticket) catch {
      _ => raise luna_coordinator_error(LunaOnlineFramedMaintenance)
    }
    let restarted = self.instance.take_restart_observation()
    let retirement_progress = self.finish_ticket_retirement(ticket, retired)
    if restarted {
      self.publish_observation(
        LunaOnlineFramedWorkerRestartObservation,
        0,
        0,
        0,
        0,
      )
      return LunaOnlineFramedCoordinatorObservationReady
    }
    return retirement_progress
  }
  if self.maintenance_kind == 4 {
    if self.tickets.is_empty() {
      self.maintenance_kind = 0
      return LunaOnlineFramedCoordinatorAdvanced
    }
    for ticket in self.tickets {
      if self.instance.multi_ticket_needs_abort(ticket) {
        self.instance.begin_abort(ticket) catch {
          _ => raise luna_coordinator_error(LunaOnlineFramedMaintenance)
        }
        return LunaOnlineFramedCoordinatorMaintenanceRequired
      }
    }
    let progress = self.instance.progress_multi_disconnect() catch {
      _ => raise luna_coordinator_error(LunaOnlineFramedMaintenance)
    }
    match progress {
      LunaOnlineInstanceTerminalizationRequired => self.maintenance_kind = 1
      LunaOnlineInstanceRequestRetirementRequired => self.maintenance_kind = 2
      LunaOnlineInstancePending
      | LunaOnlineInstanceWaiting
      | LunaOnlineInstanceEventReady => ()
    }
    return LunaOnlineFramedCoordinatorMaintenanceRequired
  }
  if self.drain_requested {
    if !self.pool.is_drained() ||
      self.queue_count != 0 ||
      self.prepared.length() != 0 {
      return LunaOnlineFramedCoordinatorDraining
    }
    if self.tickets.length() != 0 {
      return LunaOnlineFramedCoordinatorMaintenanceRequired
    }
    let shutdown = self.instance.progress_shutdown() catch {
      _ => raise luna_coordinator_error(LunaOnlineFramedMaintenance)
    }
    match shutdown {
      LunaOnlineInstanceShutdownPending => LunaOnlineFramedCoordinatorDraining
      LunaOnlineInstanceShutdownClosed => {
        self.closed = true
        LunaOnlineFramedCoordinatorClosed
      }
    }
  } else {
    LunaOnlineFramedCoordinatorIdle
  }
}
