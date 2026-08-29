///|
fn LunaOnlineInstance::advance_multi_recovery_maintenance(
  self : LunaOnlineInstance,
) -> LunaRecoveryMaintenanceStep raise LunaOnlineInstanceError {
  let lease = self.lease_owner()
  if self.maintenance_state == 6 {
    if self.drain_requested {
      lease.abandon_restart() catch {
        _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
      }
      lease.begin_recovery_maintenance(ResourceUnavailable) catch {
        _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
      }
      self.maintenance_state = 1
      return LunaRecoveryMaintenancePending
    }
    let restart = lease.progress_restart_backoff() catch {
      _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
    }
    match restart {
      OnlineWorkerRestartWaiting(_) => return LunaRecoveryMaintenancePending
      OnlineWorkerRestartForbidden => {
        lease.abandon_restart() catch {
          _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
        }
        lease.begin_recovery_maintenance(ResourceUnavailable) catch {
          _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
        }
        self.maintenance_state = 1
        return LunaRecoveryMaintenancePending
      }
      OnlineWorkerRestarted => {
        self.maintenance_state = 5
        self.restart_observation_pending = true
        return LunaRecoveryMaintenanceRestarted
      }
      OnlineWorkerRestartCleanupRequired => {
        lease.begin_restart_cleanup_maintenance() catch {
          _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
        }
        self.maintenance_state = 2
        return LunaRecoveryMaintenancePending
      }
    }
  }
  if self.maintenance_state == 5 {
    let complete = lease.progress_restarted_failure_terminals() catch {
      _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
    }
    if !complete {
      return LunaRecoveryMaintenancePending
    }
    for lane in self.multi.lanes {
      if lane.phase != LunaInstanceIdlePhase {
        lane.failure_kind = WorkerLatchedFailure
        lane.terminal_cut = WorkerFailureCut
        lane.restarted_failure_terminal_pending = true
        lane.recovery_close = false
      }
    }
    self.maintenance_state = 0
    self.recovery_close = false
    return LunaRecoveryMaintenanceRestarted
  }
  if self.maintenance_state == 0 {
    lease.begin_recovery_maintenance(ResourceUnavailable) catch {
      _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
    }
    self.maintenance_state = 1
    return LunaRecoveryMaintenancePending
  }
  let progress = lease.progress_maintenance() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
  }
  match progress {
    LunaWorkerServiceRestartReady => {
      self.maintenance_state = 6
      return LunaRecoveryMaintenancePending
    }
    LunaWorkerServiceRecoveryReady => ()
    LunaWorkerServiceFailCloseReady => {
      self.maintenance_state = 0
      self.recovery_close = true
      return LunaRecoveryMaintenanceFailCloseReady
    }
    LunaWorkerServiceClosed => {
      self.maintenance_state = 0
      self.recovery_close = true
      return LunaRecoveryMaintenanceClosed
    }
    _ => ()
  }
  LunaRecoveryMaintenancePending
}

///|
fn LunaOnlineInstance::progress_multi_terminalization(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
) -> LunaOnlineInstanceProgress raise LunaOnlineInstanceError {
  ignore(self.require_multi_ticket(ticket))
  if self.events.has_event() {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit))
  }
  if self.recovery_close {
    match self.advance_multi_recovery_maintenance() {
      LunaRecoveryMaintenancePending | LunaRecoveryMaintenanceRestarted =>
        return LunaOnlineInstanceTerminalizationRequired
      LunaRecoveryMaintenanceFailCloseReady | LunaRecoveryMaintenanceClosed =>
        return LunaOnlineInstanceTerminalizationRequired
    }
  }
  ignore(self.multi_route_one_publication())
  if self.events.has_event() {
    LunaOnlineInstanceEventReady
  } else {
    LunaOnlineInstanceTerminalizationRequired
  }
}

///|
fn LunaOnlineInstance::multi_has_worker_failure(
  self : LunaOnlineInstance,
) -> Bool {
  for lane in self.multi.lanes {
    if lane.phase != LunaInstanceIdlePhase &&
      lane.failure_kind == WorkerLatchedFailure {
      return true
    }
  }
  false
}

///|
fn LunaOnlineInstance::multi_event_failure_kind(
  self : LunaOnlineInstance,
) -> SessionLatchedFailure {
  let lane = self.multi.event_lane
  if lane < 0 || lane >= self.multi.lanes.length() {
    NoLatchedFailure
  } else {
    self.multi.lanes[lane].failure_kind
  }
}
