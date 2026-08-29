///|
pub fn LunaOnlineInstancePreparation::state(
  self : LunaOnlineInstancePreparation,
) -> LunaOnlineInstancePreparationState {
  self.state
}

///|
pub fn LunaOnlineInstancePreparation::take_ready(
  self : LunaOnlineInstancePreparation,
) -> LunaOnlineInstance raise LunaOnlineInstancePreparationError {
  if self.consumed || self.state != LunaOnlineInstanceReady {
    raise luna_preparation_failure(LunaOnlineInvalid(LunaOnlineLifecycle))
  }
  self.consumed = true
  match self.instance {
    Some(instance) => {
      self.instance = None
      instance
    }
    None => abort("luna online ready invariant")
  }
}

///|
pub fn LunaOnlineInstancePreparation::take_cleanup(
  self : LunaOnlineInstancePreparation,
) -> FailedLunaOnlineInstancePreparation raise LunaOnlineInstancePreparationError {
  if self.consumed || self.state != LunaOnlineInstanceCleanupRequired {
    raise luna_preparation_failure(LunaOnlineInvalid(LunaOnlineLifecycle))
  }
  self.consumed = true
  self.instance = None
  self.cleanup
}

///|
pub fn FailedLunaOnlineInstancePreparation::cause(
  self : FailedLunaOnlineInstancePreparation,
) -> LunaOnlineInstanceFailure {
  match self.cause_kind {
    PreparationServiceCleanupFailure => {
      if self.service_cleanup is None {
        abort("luna online cleanup preparation cause invariant")
      }
      LunaOnlineServicePreparationFailure
    }
    PreparationOwnershipFailure => LunaOnlineInvalid(LunaOnlineOwnership)
  }
}

///|
pub fn FailedLunaOnlineInstancePreparation::is_closed(
  self : FailedLunaOnlineInstancePreparation,
) -> Bool {
  self.closed
}

///|
pub fn FailedLunaOnlineInstancePreparation::retry_close(
  self : FailedLunaOnlineInstancePreparation,
) -> Unit raise LunaOnlineInstancePreparationError {
  if self.closed {
    return
  }
  match self.service_cleanup {
    Some(cleanup) => {
      cleanup.retry_close() catch {
        _ => raise luna_preparation_failure(LunaOnlineServicePreparationFailure)
      }
      if cleanup.is_closed() {
        self.service_cleanup = None
        self.closed = true
      }
    }
    None => abort("luna online cleanup invariant")
  }
}

///|
fn LunaOnlineInstance::require_owner(self : LunaOnlineInstance) -> Unit {
  if self.lease is None ||
    self.output.length() == 0 ||
    self.multi.lanes.length() == 0 {
    abort("luna online instance owner invariant")
  }
}

///|
fn LunaOnlineInstance::mark_event_published(
  self : LunaOnlineInstance,
  kind : SessionEventKind,
) -> Unit {
  if self.event_epoch == 0xffffffffffffffffUL {
    abort("luna online event epoch invariant")
  }
  self.event_epoch += 1UL
  self.event_kind = kind
  self.event_taken = false
}

///|
pub fn LunaOnlineInstance::take_event(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
) -> LunaOnlineEventCredit raise LunaOnlineInstanceError {
  self.require_active_ticket(ticket)
  self.require_owner()
  if ticket.lane >= 0 && self.multi.event_lane != ticket.lane {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit))
  }
  if !self.events.has_event() || self.event_taken {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit))
  }
  ignore(
    self.events.view() catch {
      _ =>
        raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit))
    },
  )
  self.event_taken = true
  {
    instance: self,
    lane: ticket.lane,
    request_epoch: ticket.epoch,
    event_epoch: self.event_epoch,
  }
}

///|
pub fn LunaOnlineEventCredit::view(
  self : LunaOnlineEventCredit,
) -> @luna_event.LunaEventView raise LunaOnlineInstanceError {
  let instance = self.instance
  instance.require_active_ticket({ lane: self.lane, epoch: self.request_epoch })
  instance.require_owner()
  if !instance.event_taken || self.event_epoch != instance.event_epoch {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit))
  }
  instance.events.view() catch {
    _ =>
      raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit))
  }
}

///|
fn LunaOnlineInstance::ack_event_credit(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
  event_epoch : UInt64,
) -> Unit raise LunaOnlineInstanceError {
  if ticket.lane >= 0 {
    self.ack_multi_event_credit(ticket, event_epoch)
    return
  }
  self.require_active_ticket(ticket)
  self.require_owner()
  if !self.event_taken || event_epoch != self.event_epoch {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit))
  }
  let view = self.events.view() catch {
    _ =>
      raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit))
  }
  match self.event_kind {
    AcceptedEvent => {
      if self.accepted_acked {
        raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit))
      }
      self.events.retire(view) catch {
        _ =>
          raise LunaOnlineInstanceFailed(
            LunaOnlineInvalid(LunaOnlineEventCredit),
          )
      }
      self.event_taken = false
      self.accepted_acked = true
    }
    TokenEvent => {
      self.events.retire(view) catch {
        _ =>
          raise LunaOnlineInstanceFailed(
            LunaOnlineInvalid(LunaOnlineEventCredit),
          )
      }
      self.event_taken = false
    }
    CompletionUsageEvent => {
      let usage = view.usage() catch {
        _ =>
          raise LunaOnlineInstanceFailed(
            LunaOnlineInvalid(LunaOnlineEventCredit),
          )
      }
      ignore(
        try! self.events.replace_usage_with_completed(
          usage,
          match self.finish_kind {
            FinishStopSequence => StopSequence
            FinishTokenLimit => TokenLimit
            FinishCancelled => Cancelled
            FinishPending => abort("luna online finish kind invariant")
          },
          self.output,
          tail_offset=0,
          tail_length=self.final_tail_length,
        ),
      )
      self.mark_event_published(CompletedEvent)
    }
    CompletedEvent => {
      self.events.retire(view) catch {
        _ =>
          raise LunaOnlineInstanceFailed(
            LunaOnlineInvalid(LunaOnlineEventCredit),
          )
      }
      self.event_taken = false
      self.phase = if self.recovery_close {
        LunaInstanceClosingTerminalPhase
      } else {
        LunaRequestReleaseReadyPhase
      }
    }
    FailureUsageEvent => {
      let usage = view.usage() catch {
        _ =>
          raise LunaOnlineInstanceFailed(
            LunaOnlineInvalid(LunaOnlineEventCredit),
          )
      }
      let code = match self.failure_kind {
        DeadlineLatchedFailure => self.deadline_code
        OutputLatchedFailure => self.output_code
        WorkerLatchedFailure => self.worker_code
        NoLatchedFailure => abort("luna online failure kind invariant")
      }
      ignore(
        try! self.events.replace_usage_with_failed(
          usage,
          code,
          code_offset=0,
          code_length=code.length(),
          retryable=self.failure_kind == WorkerLatchedFailure,
        ),
      )
      self.mark_event_published(FailedEvent)
    }
    FailedEvent => {
      self.events.retire(view) catch {
        _ =>
          raise LunaOnlineInstanceFailed(
            LunaOnlineInvalid(LunaOnlineEventCredit),
          )
      }
      self.event_taken = false
      self.phase = if self.recovery_close {
        LunaInstanceClosingTerminalPhase
      } else {
        LunaRequestReleaseReadyPhase
      }
    }
    NoSessionEvent => abort("luna online event kind invariant")
  }
}

///|
pub fn LunaOnlineEventCredit::ack(
  self : LunaOnlineEventCredit,
) -> Unit raise LunaOnlineInstanceError {
  self.instance.ack_event_credit(
    { lane: self.lane, epoch: self.request_epoch },
    self.event_epoch,
  )
}

///|
/// Applies the request-local cut after any pinned event authority has already
/// been revoked. A terminal event means the scheduler terminal was consumed;
/// a restarted healthy lease therefore proceeds to exact request release,
/// while a fail-close path retains its close obligation.
fn LunaOnlineInstance::finish_abort_after_event_cut(
  self : LunaOnlineInstance,
) -> Bool {
  if self.recovery_close &&
    (self.string_cut != NoStringCut || self.terminal_cut != NoTerminalCut) {
    self.phase = LunaInstanceRecoveringTerminalPhase
    return true
  }
  match self.event_kind {
    CompletionUsageEvent | CompletedEvent | FailureUsageEvent | FailedEvent => {
      self.phase = if self.recovery_close {
        LunaInstanceClosingTerminalPhase
      } else {
        LunaRequestReleaseReadyPhase
      }
      true
    }
    NoSessionEvent | AcceptedEvent | TokenEvent => false
  }
}

///|
/// Begins the exact cancellation cut once. Request and cleanup authority stay
/// authenticated by the same ticket until retirement or terminal close.
pub fn LunaOnlineInstance::begin_abort(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
) -> Unit raise LunaOnlineInstanceError {
  if ticket.lane >= 0 {
    self.begin_abort_multi(ticket)
    return
  }
  self.require_ticket(ticket)
  self.require_owner()
  if self.phase != LunaRequestActivePhase {
    return
  }
  if self.events.has_event() {
    try! self.events.discard()
    self.event_taken = false
    self.accepted_acked = true
  }
  if self.finish_abort_after_event_cut() {
    return
  }
  if self.string_cut != NoStringCut || self.terminal_cut != NoTerminalCut {
    self.phase = LunaRequestAwaitingTerminalPhase
    return
  }
  self.deferred_cancel = false
  let lease = self.lease_owner()
  let reservation = lease.reserve_cancel() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
  }
  match reservation {
    OnlineCancelReserved => {
      let committed = lease.commit_cancel() catch {
        _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
      }
      match committed {
        OnlineCancelCommitted => self.terminal_cut = CallerCancellationCut
        OnlineCancelReservationAbsent =>
          raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineLifecycle))
      }
    }
    OnlineCancelNaturalTerminal => ()
  }
  self.phase = LunaRequestAwaitingTerminalPhase
}
