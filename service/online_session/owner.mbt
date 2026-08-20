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
    self.wire_limits.inference_limits() != self.inference {
    abort("luna online instance owner invariant")
  }
}

///|
pub fn LunaOnlineInstance::has_event(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
) -> Bool raise LunaOnlineInstanceError {
  self.require_ticket(ticket)
  self.require_owner()
  self.phase == LunaRequestActivePhase && self.event_ready
}

///|
pub fn LunaOnlineInstance::event_length(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
) -> Int raise LunaOnlineInstanceError {
  self.require_active_ticket(ticket)
  self.require_owner()
  if !self.event_ready {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit))
  }
  self.writer.length() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineWireFailure)
  }
}

///|
pub fn LunaOnlineInstance::copy_event_to(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Int raise LunaOnlineInstanceError {
  self.require_active_ticket(ticket)
  self.require_owner()
  if !self.event_ready {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit))
  }
  self.writer.copy_to(destination, destination_offset~) catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineWireFailure)
  }
}

///|
pub fn LunaOnlineInstance::ack_event(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
) -> Unit raise LunaOnlineInstanceError {
  self.require_active_ticket(ticket)
  self.require_owner()
  if !self.event_ready {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit))
  }
  match self.event_kind {
    AcceptedEvent => {
      if self.accepted_acked {
        raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit))
      }
      self.writer.retire() catch {
        _ => raise LunaOnlineInstanceFailed(LunaOnlineWireFailure)
      }
      self.event_ready = false
      self.accepted_acked = true
    }
    TokenEvent => {
      self.writer.retire() catch {
        _ => raise LunaOnlineInstanceFailed(LunaOnlineWireFailure)
      }
      self.event_ready = false
    }
    CompletionUsageEvent => {
      self.writer.retire() catch {
        _ => raise LunaOnlineInstanceFailed(LunaOnlineWireFailure)
      }
      try! self.writer.prepare_completed(
        match self.finish_kind {
          FinishStopSequence => StopSequence
          FinishTokenLimit => TokenLimit
          FinishCancelled => Cancelled
          FinishPending => abort("luna online finish kind invariant")
        },
        input_tokens=self.input_tokens,
        cached_input_tokens=0,
        output_tokens=self.output_tokens,
        self.output,
        tail_offset=0,
        tail_length=self.final_tail_length,
      )
      self.event_kind = CompletedEvent
      self.event_ready = true
    }
    CompletedEvent => {
      self.writer.retire() catch {
        _ => raise LunaOnlineInstanceFailed(LunaOnlineWireFailure)
      }
      self.event_ready = false
      self.phase = if self.recovery_close {
        LunaInstanceClosingTerminalPhase
      } else {
        LunaRequestReleaseReadyPhase
      }
    }
    FailureUsageEvent => {
      self.writer.retire() catch {
        _ => raise LunaOnlineInstanceFailed(LunaOnlineWireFailure)
      }
      let code = match self.failure_kind {
        DeadlineLatchedFailure => self.deadline_code
        OutputLatchedFailure => self.output_code
        WorkerLatchedFailure => self.worker_code
        NoLatchedFailure => abort("luna online failure kind invariant")
      }
      try! self.writer.prepare_failed(
        code,
        code_offset=0,
        code_length=code.length(),
        retryable=self.failure_kind == WorkerLatchedFailure,
      )
      self.event_kind = FailedEvent
      self.event_ready = true
    }
    FailedEvent => {
      self.writer.retire() catch {
        _ => raise LunaOnlineInstanceFailed(LunaOnlineWireFailure)
      }
      self.event_ready = false
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
/// Begins the exact cancellation cut once. Request and cleanup authority stay
/// authenticated by the same ticket until retirement or terminal close.
pub fn LunaOnlineInstance::begin_abort(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
) -> Unit raise LunaOnlineInstanceError {
  self.require_ticket(ticket)
  self.require_owner()
  if self.phase != LunaRequestActivePhase {
    return
  }
  if self.event_ready {
    self.writer.retire() catch {
      _ => raise LunaOnlineInstanceFailed(LunaOnlineWireFailure)
    }
    self.event_ready = false
    self.accepted_acked = true
  }
  if self.recovery_close &&
    (self.string_cut != NoStringCut || self.terminal_cut != NoTerminalCut) {
    self.phase = LunaInstanceRecoveringTerminalPhase
    return
  }
  match self.event_kind {
    CompletionUsageEvent | CompletedEvent | FailureUsageEvent | FailedEvent => {
      self.phase = if self.recovery_close {
        LunaInstanceClosingTerminalPhase
      } else {
        LunaRequestReleaseReadyPhase
      }
      return
    }
    NoSessionEvent | AcceptedEvent | TokenEvent => ()
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
