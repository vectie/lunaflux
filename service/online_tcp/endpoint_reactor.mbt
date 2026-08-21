///|
fn LunaOnlineTcpEndpoint::capture_rejection(
  self : LunaOnlineTcpEndpoint,
) -> Bool {
  let credit = self.coordinator().take_rejection() catch {
      _ => {
        self.begin_disconnect(failure=LunaOnlineTcpCoordinator)
        return false
      }
    }
  let sequence = credit.sequence() catch {
    _ => {
      self.begin_disconnect(failure=LunaOnlineTcpCoordinator)
      return false
    }
  }
  let rule = credit.rule() catch {
    _ => {
      self.begin_disconnect(failure=LunaOnlineTcpCoordinator)
      return false
    }
  }
  credit.ack() catch {
    _ => {
      self.begin_disconnect(failure=LunaOnlineTcpCoordinator)
      return false
    }
  }
  self.rejection_sequence = sequence
  self.rejection_rule = luna_online_tcp_rejection_code(rule)
  self.rejection_latched = true
  self.begin_disconnect()
  true
}

///|
fn LunaOnlineTcpEndpoint::handle_rejection_on_reactor(
  self : LunaOnlineTcpEndpoint,
) -> LunaOnlineTcpProgress {
  let captured = self.capture_rejection()
  self.close_transport_on_reactor()
  if captured {
    LunaOnlineTcpRejected
  } else {
    LunaOnlineTcpCleanupRequired
  }
}

///|
fn luna_online_tcp_map_coordinator_progress(
  endpoint : LunaOnlineTcpEndpoint,
  progress : @online_session.LunaOnlineFramedCoordinatorProgress,
) -> LunaOnlineTcpProgress {
  match progress {
    LunaOnlineFramedCoordinatorIdle =>
      if endpoint.drain_requested || endpoint.disconnect_requested {
        LunaOnlineTcpCleanupRequired
      } else {
        LunaOnlineTcpBackpressured
      }
    LunaOnlineFramedCoordinatorAwaitingInput => LunaOnlineTcpBackpressured
    LunaOnlineFramedCoordinatorAdvanced
    | LunaOnlineFramedCoordinatorEventPreparing
    | LunaOnlineFramedCoordinatorEventReady => LunaOnlineTcpAdvanced
    LunaOnlineFramedCoordinatorRejectionReady =>
      if endpoint.capture_rejection() {
        LunaOnlineTcpRejected
      } else {
        LunaOnlineTcpCleanupRequired
      }
    LunaOnlineFramedCoordinatorMaintenanceRequired => {
      endpoint.maintenance_handoff.arm()
      LunaOnlineTcpMaintenanceRequired
    }
    LunaOnlineFramedCoordinatorDraining => {
      endpoint.drain_requested = true
      endpoint.phase = LUNA_ONLINE_TCP_DRAINING
      LunaOnlineTcpCleanupRequired
    }
    LunaOnlineFramedCoordinatorClosed => {
      endpoint.close_pending = endpoint.servers.length() == 1 ||
        endpoint.connections.length() == 1
      endpoint.phase = if endpoint.close_pending {
        LUNA_ONLINE_TCP_CLOSE_REQUIRED
      } else {
        LUNA_ONLINE_TCP_CLOSED
      }
      endpoint.maintenance_handoff.clear()
      if endpoint.close_pending {
        LunaOnlineTcpCleanupRequired
      } else {
        LunaOnlineTcpClosedProgress
      }
    }
  }
}

///|
pub async fn LunaOnlineTcpEndpoint::progress_on_reactor(
  self : LunaOnlineTcpEndpoint,
) -> LunaOnlineTcpProgress {
  self.activity.enter_reactor()
  defer self.activity.leave_reactor()
  if self.phase == LUNA_ONLINE_TCP_CLOSED {
    return LunaOnlineTcpClosedProgress
  }
  if self.phase == LUNA_ONLINE_TCP_LISTENING ||
    self.phase == LUNA_ONLINE_TCP_ACCEPTING {
    return self.accept_on_reactor()
  }
  let mut transitions = 0
  while transitions < self.limits.reactor_transition_budget {
    if self.close_pending {
      self.close_transport_on_reactor()
      transitions += 1
      if self.publish_closed_if_ready() {
        return LunaOnlineTcpClosedProgress
      }
      if transitions >= self.limits.reactor_transition_budget {
        return LunaOnlineTcpCleanupRequired
      }
    }
    if self.maintenance_handoff.is_pending() {
      return LunaOnlineTcpMaintenanceRequired
    }
    if self.tail_length > 0 &&
      !self.tail_backpressured &&
      !self.drain_requested &&
      !self.disconnect_requested {
      transitions += 1
      let offered = self.offer_tail()
      match offered {
        LunaOnlineTcpCleanupRequired | LunaOnlineTcpRejected => return offered
        LunaOnlineTcpBackpressured =>
          if transitions >= self.limits.reactor_transition_budget {
            return LunaOnlineTcpBackpressured
          }
        _ => continue
      }
    }
    let progress = self.coordinator().progress() catch {
        _ => {
          self.begin_disconnect(failure=LunaOnlineTcpCoordinator)
          self.close_transport_on_reactor()
          return LunaOnlineTcpCleanupRequired
        }
      }
    transitions += 1
    self.tail_backpressured = false
    match progress {
      LunaOnlineFramedCoordinatorIdle => {
        if self.drain_requested || self.disconnect_requested {
          self.phase = LUNA_ONLINE_TCP_DRAINING
          return LunaOnlineTcpCleanupRequired
        }
        if self.tail_length > 0 {
          continue
        }
        return self.read_on_reactor()
      }
      LunaOnlineFramedCoordinatorAwaitingInput => {
        if self.drain_requested || self.disconnect_requested {
          self.phase = LUNA_ONLINE_TCP_DRAINING
          return LunaOnlineTcpCleanupRequired
        }
        if self.tail_length > 0 {
          continue
        }
        return self.read_on_reactor()
      }
      LunaOnlineFramedCoordinatorAdvanced
      | LunaOnlineFramedCoordinatorEventPreparing => continue
      LunaOnlineFramedCoordinatorEventReady =>
        return self.write_event_on_reactor()
      LunaOnlineFramedCoordinatorRejectionReady =>
        return self.handle_rejection_on_reactor()
      LunaOnlineFramedCoordinatorMaintenanceRequired => {
        self.maintenance_handoff.arm()
        return LunaOnlineTcpMaintenanceRequired
      }
      LunaOnlineFramedCoordinatorDraining => {
        self.drain_requested = true
        self.phase = LUNA_ONLINE_TCP_DRAINING
        return LunaOnlineTcpCleanupRequired
      }
      LunaOnlineFramedCoordinatorClosed => {
        self.close_pending = self.servers.length() == 1 ||
          self.connections.length() == 1
        if self.close_pending {
          self.phase = LUNA_ONLINE_TCP_CLOSE_REQUIRED
          self.close_transport_on_reactor()
        }
        self.phase = LUNA_ONLINE_TCP_CLOSED
        self.maintenance_handoff.clear()
        return LunaOnlineTcpClosedProgress
      }
    }
  }
  LunaOnlineTcpAdvanced
}

///|
pub fn LunaOnlineTcpEndpoint::progress_off_reactor_maintenance(
  self : LunaOnlineTcpEndpoint,
) -> LunaOnlineTcpProgress raise LunaOnlineTcpError {
  self.activity.enter_maintenance()
  defer self.activity.leave_maintenance()
  if self.phase == LUNA_ONLINE_TCP_CLOSED {
    return LunaOnlineTcpClosedProgress
  }
  self.maintenance_handoff.require_pending()
  if self.close_pending {
    return LunaOnlineTcpCleanupRequired
  }
  let progress = self.coordinator().progress_off_reactor_maintenance() catch {
      _ => {
        self.begin_disconnect(failure=LunaOnlineTcpCoordinator)
        return LunaOnlineTcpCleanupRequired
      }
    }
  if progress != LunaOnlineFramedCoordinatorMaintenanceRequired {
    self.maintenance_handoff.clear()
  }
  if progress == LunaOnlineFramedCoordinatorRejectionReady {
    let captured = self.capture_rejection()
    return if captured {
      LunaOnlineTcpRejected
    } else {
      LunaOnlineTcpCleanupRequired
    }
  }
  luna_online_tcp_map_coordinator_progress(self, progress)
}
