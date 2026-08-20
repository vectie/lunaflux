///|
fn deadline_failure_code() -> FixedArray[Byte] {
  let source = b"deadline_exceeded"
  FixedArray::makei(source.length(), index => source[index])
}

///|
fn output_failure_code() -> FixedArray[Byte] {
  let source = b"output_invalid"
  FixedArray::makei(source.length(), index => source[index])
}

///|
fn worker_failure_code() -> FixedArray[Byte] {
  let source = b"worker_unavailable"
  FixedArray::makei(source.length(), index => source[index])
}

///|
fn LunaOnlineInstance::begin_caller_cut(
  self : LunaOnlineInstance,
  lease : @worker_service.OnlineWorkerLease,
) -> Unit raise LunaOnlineInstanceError {
  let reservation = lease.reserve_cancel() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
  }
  match reservation {
    OnlineCancelNaturalTerminal => ()
    OnlineCancelReserved => {
      let committed = try! lease.commit_cancel()
      match committed {
        OnlineCancelCommitted => self.terminal_cut = CallerCancellationCut
        OnlineCancelReservationAbsent => abort("online caller cut invariant")
      }
    }
  }
}

///|
fn LunaOnlineInstance::begin_deadline_cut(
  self : LunaOnlineInstance,
  lease : @worker_service.OnlineWorkerLease,
) -> Unit raise LunaOnlineInstanceError {
  let expiry = lease.expire() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
  }
  match expiry {
    OnlineDeadlinePending | OnlineDeadlineNaturalTerminal => ()
    OnlineDeadlineCommitted => {
      self.failure_kind = DeadlineLatchedFailure
      self.terminal_cut = DeadlineFailureCut
    }
  }
}

///|
fn LunaOnlineInstance::enforce_deadline(
  self : LunaOnlineInstance,
  lease : @worker_service.OnlineWorkerLease,
) -> Bool raise LunaOnlineInstanceError {
  let expiry = lease.expire() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
  }
  match classify_expiry_cut(expiry) {
    ExpiryCutCommitted => {
      self.failure_kind = DeadlineLatchedFailure
      self.terminal_cut = DeadlineFailureCut
      true
    }
    ExpiryCutPending | ExpiryCutNaturalTerminal => false
  }
}

///|
fn classify_expiry_cut(
  expiry : @worker_service.OnlineExpiry,
) -> ExpiryCutClassification {
  match expiry {
    OnlineDeadlinePending => ExpiryCutPending
    OnlineDeadlineNaturalTerminal => ExpiryCutNaturalTerminal
    OnlineDeadlineCommitted => ExpiryCutCommitted
  }
}

///|
/// Latches only failure classes whose lower owner has already retained every
/// outstanding flight/recovery obligation. Retryable pre-mutation failures do
/// not consume the session or change its public terminal cause.
fn LunaOnlineInstance::latch_worker_step_failure(
  self : LunaOnlineInstance,
  step : @worker_service.OnlineWorkerStep,
) -> Bool {
  match step {
    OnlineWorkerOutputInvalid => {
      if self.terminal_cut == NoTerminalCut && self.string_cut == NoStringCut {
        if self.failure_kind == NoLatchedFailure {
          self.failure_kind = OutputLatchedFailure
        }
        self.terminal_cut = WorkerFailureCut
      }
      self.recovery_close = true
      true
    }
    OnlineWorkerUnavailable => {
      if self.terminal_cut == NoTerminalCut && self.string_cut == NoStringCut {
        if self.failure_kind == NoLatchedFailure {
          self.failure_kind = WorkerLatchedFailure
        }
        self.terminal_cut = WorkerFailureCut
      }
      self.recovery_close = true
      true
    }
    OnlineWorkerAdvanced | OnlineWorkerRetryable => false
  }
}

///|
fn LunaOnlineInstance::prepare_cut_terminal(self : LunaOnlineInstance) -> Unit {
  try! self.events.publish_usage(
    input_tokens=self.input_tokens,
    cached_input_tokens=0,
    output_tokens=self.output_tokens,
  )
  let kind = if self.terminal_cut == CallerCancellationCut {
    self.finish_kind = FinishCancelled
    CompletionUsageEvent
  } else {
    if self.failure_kind == NoLatchedFailure {
      self.failure_kind = match self.terminal_cut {
        DeadlineFailureCut => DeadlineLatchedFailure
        WorkerFailureCut => WorkerLatchedFailure
        OutputCancellationCut | NaturalOutputFailureCut => OutputLatchedFailure
        NoTerminalCut | CallerCancellationCut =>
          abort("online terminal failure kind invariant")
      }
    }
    FailureUsageEvent
  }
  self.terminal_cut = NoTerminalCut
  self.mark_event_published(kind)
}

///|
fn LunaOnlineInstance::progress_terminal_cut(
  self : LunaOnlineInstance,
  lease : @worker_service.OnlineWorkerLease,
) -> Unit {
  if !(try! lease.has_publication()) {
    let step = try! lease.progress_status()
    if self.latch_worker_step_failure(step) {
      return
    }
    return
  }
  let publication = try! lease.take_publication()
  match publication.kind() {
    OnlineSuppressedTokenPublication => self.output_tokens += 1
    OnlineGeneratedTokenPublication =>
      abort("online terminal cut token suppression invariant")
    OnlineTerminalPublication => {
      let reason = try! publication.terminal_reason()
      if self.terminal_cut == CallerCancellationCut {
        if reason != OnlineCancelledByCaller {
          abort("online caller cancellation terminal invariant")
        }
      } else if self.terminal_cut == DeadlineFailureCut &&
        reason != OnlineDeadlineReached {
        abort("online deadline terminal invariant")
      } else if self.terminal_cut == OutputCancellationCut &&
        reason != OnlineCancelledByCaller {
        abort("online output cancellation terminal invariant")
      } else if self.terminal_cut == WorkerFailureCut &&
        reason != OnlineWorkerFailed {
        abort("online worker failure terminal invariant")
      } else if self.terminal_cut == NaturalOutputFailureCut &&
        reason != OnlineMaximumOutputReached &&
        reason != OnlineStopTokenReached {
        abort("online natural output failure terminal invariant")
      }
      self.prepare_cut_terminal()
    }
  }
}

///|
/// Advances one explicitly off-reactor failure-terminalization transition.
/// This is the only normal-session API allowed to recover/reap a failed child.
/// The first latched product cause remains stable across recovery retries.
pub fn LunaOnlineInstance::progress_terminalization(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
) -> LunaOnlineInstanceProgress raise LunaOnlineInstanceError {
  self.require_ticket(ticket)
  if self.events.has_event() {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit))
  }
  if self.phase != LunaRequestActivePhase || !self.accepted_acked {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineLifecycle))
  }
  self.require_owner()
  if !self.recovery_close && self.terminal_cut != OutputCancellationCut {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineLifecycle))
  }
  let lease = self.lease_owner()
  if self.recovery_close {
    let recovery = lease.progress_terminal_recovery_status(ResourceUnavailable)
    match recovery {
      OnlineTerminalRecoveryRetryable | OnlineTerminalRecoveryStillPending =>
        return LunaOnlineInstanceTerminalizationRequired
      OnlineTerminalRecoveryReady => ()
    }
  } else if !(try! lease.has_publication()) {
    let step = try! lease.progress_status()
    ignore(self.latch_worker_step_failure(step))
    return LunaOnlineInstanceTerminalizationRequired
  }
  if !(try! lease.has_publication()) {
    return LunaOnlineInstanceTerminalizationRequired
  }
  if self.string_cut != NoStringCut {
    self.progress_string_cut(lease)
  } else {
    self.progress_terminal_cut(lease)
  }
  if self.events.has_event() {
    LunaOnlineInstanceEventReady
  } else {
    LunaOnlineInstanceTerminalizationRequired
  }
}

///|
/// Requests a caller-visible cancellation. A pinned Token remains immutable;
/// its acknowledgement is required before the exact cut can begin.
pub fn LunaOnlineInstance::request_cancel(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
) -> Unit raise LunaOnlineInstanceError {
  self.require_active_ticket(ticket)
  self.require_owner()
  if self.terminal_cut != NoTerminalCut || self.string_cut != NoStringCut {
    return
  }
  if self.deferred_cancel {
    return
  }
  match self.event_kind {
    CompletionUsageEvent | CompletedEvent | FailureUsageEvent | FailedEvent =>
      return
    NoSessionEvent | AcceptedEvent | TokenEvent => ()
  }
  if self.events.has_event() {
    match self.event_kind {
      AcceptedEvent | TokenEvent => self.deferred_cancel = true
      _ =>
        raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit))
    }
    return
  }
  let lease = self.lease_owner()
  if self.enforce_deadline(lease) {
    return
  }
  self.begin_caller_cut(lease)
}

///|
/// Samples the owner-bound monotonic clock only after event credit is free.
/// Natural terminal evidence takes precedence inside the lower lease.
pub fn LunaOnlineInstance::check_deadline(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
) -> Unit raise LunaOnlineInstanceError {
  self.require_active_ticket(ticket)
  self.require_owner()
  if self.terminal_cut != NoTerminalCut || self.string_cut != NoStringCut {
    return
  }
  if self.deferred_cancel {
    return
  }
  match self.event_kind {
    CompletionUsageEvent | CompletedEvent | FailureUsageEvent | FailedEvent =>
      return
    NoSessionEvent | AcceptedEvent | TokenEvent => ()
  }
  if self.events.has_event() {
    // Poll only: never sample time or latch deadline intent while a frame is
    // pinned. Automatic expiry runs from the next credit-free progress call.
    return
  }
  self.begin_deadline_cut(self.lease_owner())
}
