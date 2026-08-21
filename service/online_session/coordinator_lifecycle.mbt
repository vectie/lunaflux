///|
pub fn LunaOnlineFramedCoordinator::request_cancel(
  self : LunaOnlineFramedCoordinator,
) -> Unit raise LunaOnlineFramedCoordinatorError {
  if self.tickets.length() != 1 || self.disconnect_requested {
    raise luna_coordinator_error(LunaOnlineFramedLifecycle)
  }
  self.instance.request_cancel(self.ticket()) catch {
    _ => raise luna_coordinator_error(LunaOnlineFramedMaintenance)
  }
}

///|
/// Cuts the single transport-neutral stream. Framed authority is revoked
/// before semantic event authority, and all later cleanup remains owned by the
/// off-reactor maintenance path.
pub fn LunaOnlineFramedCoordinator::disconnect(
  self : LunaOnlineFramedCoordinator,
) -> Unit raise LunaOnlineFramedCoordinatorError {
  if self.closed {
    return
  }
  self.drain_requested = true
  self.disconnect_requested = true
  self.pool.begin_drain()
  self.instance.begin_drain()
  if self.rejection_epoch != 0UL {
    self.rejection_epoch = 0UL
    self.rejection_sequence = 0UL
    self.rejection_rule = 0
    self.rejection_taken = false
    self.stall_active = false
    self.stall_deadline_millis = 0UL
  }
  if self.tickets.length() == 1 {
    self.begin_output_disconnect()
  } else {
    self.release_outbound_authority()
    if self.event_credits.length() != 0 {
      raise luna_coordinator_error(LunaOnlineFramedMaintenance)
    }
  }
}

///|
pub fn LunaOnlineFramedCoordinator::begin_drain(
  self : LunaOnlineFramedCoordinator,
) -> Unit {
  if self.closed || self.drain_requested {
    return
  }
  self.drain_requested = true
  self.pool.begin_drain()
  self.instance.begin_drain()
}

///|
fn LunaOnlineFramedCoordinator::finish_ticket_retirement(
  self : LunaOnlineFramedCoordinator,
  progress : LunaOnlineInstanceRetirementProgress,
) -> LunaOnlineFramedCoordinatorProgress {
  match progress {
    LunaOnlineInstanceRetirementPending =>
      LunaOnlineFramedCoordinatorMaintenanceRequired
    LunaOnlineInstanceIdle => {
      self.tickets.clear()
      self.maintenance_kind = 0
      self.disconnect_requested = false
      if self.drain_requested {
        LunaOnlineFramedCoordinatorDraining
      } else {
        LunaOnlineFramedCoordinatorAdvanced
      }
    }
    LunaOnlineInstanceClosed => {
      self.tickets.clear()
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
pub fn LunaOnlineFramedCoordinator::progress_off_reactor_maintenance(
  self : LunaOnlineFramedCoordinator,
) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError {
  if self.closed {
    return LunaOnlineFramedCoordinatorClosed
  }
  if self.rejection_epoch != 0UL {
    if !self.enforce_rejection_stall() {
      return LunaOnlineFramedCoordinatorRejectionReady
    }
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
    self.semantic_event_pending {
    return LunaOnlineFramedCoordinatorEventPreparing
  }
  if self.disconnect_requested && self.tickets.length() == 1 {
    self.begin_output_disconnect() catch {
      _ => return LunaOnlineFramedCoordinatorMaintenanceRequired
    }
  }
  if self.maintenance_kind == 1 {
    if self.tickets.length() != 1 {
      raise luna_coordinator_error(LunaOnlineFramedLifecycle)
    }
    let progress = self.instance.progress_terminalization(self.ticket()) catch {
      _ => raise luna_coordinator_error(LunaOnlineFramedMaintenance)
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
      LunaOnlineInstancePending =>
        return LunaOnlineFramedCoordinatorMaintenanceRequired
    }
  }
  if self.maintenance_kind == 2 {
    if self.tickets.length() != 1 {
      raise luna_coordinator_error(LunaOnlineFramedLifecycle)
    }
    let retired = self.instance.progress_request_retirement(self.ticket()) catch {
      _ => raise luna_coordinator_error(LunaOnlineFramedMaintenance)
    }
    return self.finish_ticket_retirement(retired)
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

///|
pub fn LunaOnlineFramedCoordinator::is_closed(
  self : LunaOnlineFramedCoordinator,
) -> Bool {
  self.closed
}
