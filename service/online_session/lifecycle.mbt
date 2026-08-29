///|
fn LunaOnlineInstance::reset_request(self : LunaOnlineInstance) -> Unit {
  if self.claim is Some(_) {
    abort("luna online request reset retained prepared claim")
  }
  self.input_tokens = 0
  self.cached_input_tokens = 0
  self.event_kind = NoSessionEvent
  self.event_taken = false
  self.accepted_acked = false
  self.output_tokens = 0
  self.final_tail_length = 0
  self.finish_kind = FinishPending
  self.string_cut = NoStringCut
  self.terminal_cut = NoTerminalCut
  self.deferred_cancel = false
  self.failure_kind = NoLatchedFailure
  self.recovery_close = false
  self.maintenance_state = 0
  self.restart_observation_pending = false
  self.restarted_failure_terminal_pending = false
  for index in 0..<self.output.length() {
    self.output[index] = b'\x00'
  }
}

///|
/// Releases only after lower scheduler authority is proven absent. A stale
/// claim here is an owner-internal invariant violation, not a retryable service
/// outcome: lower retirement/close has already committed.
fn LunaOnlineInstance::release_request_claim(self : LunaOnlineInstance) -> Unit {
  let claim = self.claim_owner()
  try! claim.release()
  self.claim = None
}

///|
fn LunaOnlineInstance::close_terminal_owner(
  self : LunaOnlineInstance,
) -> LunaOnlineInstanceRetirementProgress raise LunaOnlineInstanceError {
  let lease = self.lease_owner()
  if self.maintenance_state == 0 {
    lease.begin_shutdown_maintenance() catch {
      _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
    }
    self.maintenance_state = 3
    self.phase = LunaInstanceRetryingTerminalClosePhase
    return LunaOnlineInstanceRetirementPending
  }
  if self.maintenance_state != 3 {
    abort("luna online terminal close maintenance invariant")
  }
  let progress = lease.progress_maintenance() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
  }
  match progress {
    LunaWorkerServiceClosed => ()
    _ => return LunaOnlineInstanceRetirementPending
  }
  self.release_request_claim()
  self.lease = None
  self.reset_request()
  self.drain_requested = true
  self.phase = LunaInstanceClosedPhase
  LunaOnlineInstanceClosed
}

///|
fn LunaOnlineInstance::advance_recovery_maintenance(
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
        lease.prepare_restarted_failure_terminal() catch {
          _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
        }
        self.maintenance_state = 0
        self.recovery_close = false
        self.restarted_failure_terminal_pending = true
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
    // Child cleanup completed, but scheduler/flight retirement remains under
    // the same cooperative recovery owner.
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
/// Returns a bounded host wake interval without sleeping or advancing restart.
pub fn LunaOnlineInstance::maintenance_wait_remaining_millis(
  self : LunaOnlineInstance,
) -> Int raise LunaOnlineInstanceError {
  self.require_owner()
  if self.maintenance_state == 6 {
    return self.lease_owner().restart_backoff_remaining_millis() catch {
        _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
      }
  }
  if self.maintenance_state == 0 || self.maintenance_state == 5 {
    return 0
  }
  self.lease_owner().maintenance_remaining_millis() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
  }
}

///|
fn LunaOnlineInstance::progress_recovery_maintenance(
  self : LunaOnlineInstance,
) -> LunaOnlineInstanceRetirementProgress raise LunaOnlineInstanceError {
  match self.advance_recovery_maintenance() {
    LunaRecoveryMaintenanceRestarted =>
      self.phase = LunaRequestAwaitingTerminalPhase
    LunaRecoveryMaintenanceFailCloseReady | LunaRecoveryMaintenanceClosed => {
      self.recovery_close = true
      self.phase = LunaRequestAwaitingTerminalPhase
    }
    LunaRecoveryMaintenancePending => ()
  }
  LunaOnlineInstanceRetirementPending
}

///|
fn LunaOnlineInstance::take_restart_observation(
  self : LunaOnlineInstance,
) -> Bool {
  if !self.restart_observation_pending {
    return false
  }
  self.restart_observation_pending = false
  true
}

///|
fn LunaOnlineInstance::authenticate_retirement_terminal(
  self : LunaOnlineInstance,
  publication : @worker_service.OnlineWorkerPublication,
) -> Unit {
  let reason = try! publication.terminal_reason()
  if self.string_cut == CancelledStringCut {
    if reason != OnlineCancelledByCaller {
      abort("luna online abort string-cut terminal invariant")
    }
    return
  }
  if self.string_cut == NaturalStringCut {
    if reason != OnlineMaximumOutputReached && reason != OnlineStopTokenReached {
      abort("luna online abort natural-string terminal invariant")
    }
    return
  }
  match self.terminal_cut {
    CallerCancellationCut | OutputCancellationCut =>
      if reason != OnlineCancelledByCaller {
        abort("luna online abort cancellation terminal invariant")
      }
    DeadlineFailureCut =>
      if reason != OnlineDeadlineReached {
        abort("luna online abort deadline terminal invariant")
      }
    WorkerFailureCut =>
      if reason != OnlineWorkerFailed {
        abort("luna online abort worker terminal invariant")
      }
    NaturalOutputFailureCut =>
      if reason != OnlineMaximumOutputReached &&
        reason != OnlineStopTokenReached {
        abort("luna online abort output terminal invariant")
      }
    NoTerminalCut =>
      if reason == OnlineWorkerFailed {
        self.recovery_close = true
      } else if reason != OnlineMaximumOutputReached &&
        reason != OnlineStopTokenReached {
        abort("luna online abort natural terminal invariant")
      }
  }
}

///|
/// Advances one request-retirement transition. Healthy terminal authority is
/// returned to the same lease without resetting its global publication cursor;
/// failed worker/device authority follows the close-only recovery path.
pub fn LunaOnlineInstance::progress_request_retirement(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
) -> LunaOnlineInstanceRetirementProgress raise LunaOnlineInstanceError {
  if ticket.lane >= 0 {
    return self.progress_multi_retirement(ticket)
  }
  self.require_ticket(ticket)
  self.require_owner()
  let lease = self.lease_owner()
  if self.phase == LunaRequestAwaitingTerminalPhase {
    let available = lease.has_publication() catch {
      _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
    }
    if !available {
      try lease.progress() catch {
        _ => {
          self.phase = LunaInstanceRecoveringTerminalPhase
          self.recovery_close = true
        }
      } noraise {
        _ => ()
      }
      return LunaOnlineInstanceRetirementPending
    }
    let publication = lease.take_publication() catch {
      _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
    }
    match publication.kind() {
      OnlineTerminalPublication => {
        self.authenticate_retirement_terminal(publication)
        self.phase = if self.recovery_close {
          LunaInstanceClosingTerminalPhase
        } else {
          LunaRequestReleaseReadyPhase
        }
      }
      // Abort retirement has no caller-visible Usage bundle. These exact
      // publications are consumed only to reach the authenticated terminal;
      // they deliberately do not advance request output counters.
      OnlineSuppressedTokenPublication | OnlineGeneratedTokenPublication => ()
    }
    return LunaOnlineInstanceRetirementPending
  }
  if self.phase == LunaInstanceRecoveringTerminalPhase {
    return self.progress_recovery_maintenance()
  }
  if self.phase == LunaInstanceClosingTerminalPhase ||
    self.phase == LunaInstanceRetryingTerminalClosePhase {
    return self.close_terminal_owner()
  }
  if self.phase != LunaRequestReleaseReadyPhase || self.recovery_close {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineLifecycle))
  }
  lease.retire_terminal_request() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
  }
  self.release_request_claim()
  self.reset_request()
  self.phase = LunaInstanceIdlePhase
  if self.drain_requested {
    self.phase = LunaInstanceDrainingPhase
  }
  LunaOnlineInstanceIdle
}

///|
/// Prevents new admissions. An active request retains its ticket and may drain
/// normally; an idle instance becomes immediately eligible for shutdown.
pub fn LunaOnlineInstance::begin_drain(self : LunaOnlineInstance) -> Unit {
  if self.phase == LunaInstanceClosedPhase {
    return
  }
  self.require_owner()
  self.drain_requested = true
  if self.phase == LunaInstanceIdlePhase {
    self.phase = LunaInstanceDrainingPhase
  }
}

///|
/// Advances explicit instance shutdown. This is the only healthy path that
/// closes the persistent empty lease and rooted worker.
pub fn LunaOnlineInstance::progress_shutdown(
  self : LunaOnlineInstance,
) -> LunaOnlineInstanceShutdownProgress raise LunaOnlineInstanceError {
  if self.phase == LunaInstanceClosedPhase {
    return LunaOnlineInstanceShutdownClosed
  }
  if !self.drain_requested {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineLifecycle))
  }
  if self.phase != LunaInstanceDrainingPhase &&
    self.phase != LunaInstanceRetryingEmptyClosePhase {
    return LunaOnlineInstanceShutdownPending
  }
  let lease = self.lease_owner()
  if self.maintenance_state == 0 {
    lease.begin_shutdown_maintenance() catch {
      _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
    }
    self.maintenance_state = 3
    self.phase = LunaInstanceRetryingEmptyClosePhase
    return LunaOnlineInstanceShutdownPending
  }
  if self.maintenance_state != 3 {
    abort("luna online empty close maintenance invariant")
  }
  let progress = lease.progress_maintenance() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
  }
  if progress != LunaWorkerServiceClosed {
    return LunaOnlineInstanceShutdownPending
  }
  self.lease = None
  self.phase = LunaInstanceClosedPhase
  LunaOnlineInstanceShutdownClosed
}
