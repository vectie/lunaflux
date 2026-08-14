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
fn OnlineSession::begin_caller_cut(
  self : OnlineSession,
  lease : @worker_service.OnlineWorkerLease,
) -> Unit raise OnlineSessionError {
  let reservation = lease.reserve_cancel() catch {
    _ => raise SessionFailed(ServiceFailure)
  }
  match reservation {
    OnlineCancelNaturalTerminal => ()
    OnlineCancelReserved => {
      let committed = try! lease.commit_cancel()
      match committed {
        OnlineCancelCommitted => self.terminal_cut = 1
        OnlineCancelReservationAbsent => abort("online caller cut invariant")
      }
    }
  }
}

///|
fn OnlineSession::begin_deadline_cut(
  self : OnlineSession,
  lease : @worker_service.OnlineWorkerLease,
) -> Unit raise OnlineSessionError {
  let expiry = lease.expire() catch { _ => raise SessionFailed(ServiceFailure) }
  match expiry {
    OnlineDeadlinePending | OnlineDeadlineNaturalTerminal => ()
    OnlineDeadlineCommitted => {
      self.failure_kind = 1
      self.terminal_cut = 2
    }
  }
}

///|
fn OnlineSession::enforce_deadline(
  self : OnlineSession,
  lease : @worker_service.OnlineWorkerLease,
) -> Bool raise OnlineSessionError {
  let expiry = lease.expire() catch { _ => raise SessionFailed(ServiceFailure) }
  match classify_expiry_cut(expiry) {
    2 => {
      self.failure_kind = 1
      self.terminal_cut = 2
      true
    }
    _ => false
  }
}

///|
fn classify_expiry_cut(expiry : @worker_service.OnlineExpiry) -> Int {
  match expiry {
    OnlineDeadlinePending => 0
    OnlineDeadlineNaturalTerminal => 1
    OnlineDeadlineCommitted => 2
  }
}

///|
/// Latches only failure classes whose lower owner has already retained every
/// outstanding flight/recovery obligation. Retryable pre-mutation failures do
/// not consume the session or change its public terminal cause.
fn OnlineSession::latch_worker_step_failure(
  self : OnlineSession,
  step : @worker_service.OnlineWorkerStep,
) -> Bool {
  match step {
    OnlineWorkerOutputInvalid => {
      if self.terminal_cut == 0 && self.string_cut == 0 {
        if self.failure_kind == 0 {
          self.failure_kind = 2
        }
        self.terminal_cut = 4
      }
      self.recovery_close = true
      true
    }
    OnlineWorkerUnavailable => {
      if self.terminal_cut == 0 && self.string_cut == 0 {
        if self.failure_kind == 0 {
          self.failure_kind = 3
        }
        self.terminal_cut = 4
      }
      self.recovery_close = true
      true
    }
    OnlineWorkerAdvanced | OnlineWorkerRetryable => false
  }
}

///|
fn OnlineSession::prepare_cut_terminal(self : OnlineSession) -> Unit {
  try! self.writer.prepare_usage(
    input_tokens=self.input_tokens,
    cached_input_tokens=0,
    output_tokens=self.output_tokens,
  )
  if self.terminal_cut == 1 {
    self.finish_kind = 3
    self.event_kind = 3
  } else {
    if self.failure_kind == 0 {
      self.failure_kind = if self.terminal_cut == 2 {
        1
      } else if self.terminal_cut == 4 {
        3
      } else {
        2
      }
    }
    self.event_kind = 5
  }
  self.terminal_cut = 0
  self.event_ready = true
}

///|
fn OnlineSession::progress_terminal_cut(
  self : OnlineSession,
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
      if self.terminal_cut == 1 {
        if reason != OnlineCancelledByCaller {
          abort("online caller cancellation terminal invariant")
        }
      } else if self.terminal_cut == 2 && reason != OnlineDeadlineReached {
        abort("online deadline terminal invariant")
      } else if self.terminal_cut == 3 && reason != OnlineCancelledByCaller {
        abort("online output cancellation terminal invariant")
      } else if self.terminal_cut == 4 && reason != OnlineWorkerFailed {
        abort("online worker failure terminal invariant")
      } else if self.terminal_cut == 5 &&
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
pub fn OnlineSession::progress_terminalization(
  self : OnlineSession,
) -> OnlineSessionProgress raise OnlineSessionError {
  if self.event_ready {
    raise SessionFailed(Invalid(EventCredit))
  }
  if self.closed || self.cleanup_phase != 0 || !self.accepted_acked {
    raise SessionFailed(Invalid(Lifecycle))
  }
  self.require_owner()
  if !self.recovery_close && self.terminal_cut != 3 {
    raise SessionFailed(Invalid(Lifecycle))
  }
  let lease = self.lease_owner()
  if self.recovery_close {
    let recovery = lease.progress_terminal_recovery_status(ResourceUnavailable)
    match recovery {
      OnlineTerminalRecoveryRetryable | OnlineTerminalRecoveryStillPending =>
        return OnlineSessionTerminalizationRequired
      OnlineTerminalRecoveryReady => ()
    }
  } else if !(try! lease.has_publication()) {
    let step = try! lease.progress_status()
    ignore(self.latch_worker_step_failure(step))
    return OnlineSessionTerminalizationRequired
  }
  if !(try! lease.has_publication()) {
    return OnlineSessionTerminalizationRequired
  }
  if self.string_cut != 0 {
    self.progress_string_cut(lease)
  } else {
    self.progress_terminal_cut(lease)
  }
  if self.event_ready {
    OnlineSessionEventReady
  } else {
    OnlineSessionTerminalizationRequired
  }
}

///|
/// Requests a caller-visible cancellation. A pinned Token remains immutable;
/// its acknowledgement is required before the exact cut can begin.
pub fn OnlineSession::request_cancel(
  self : OnlineSession,
) -> Unit raise OnlineSessionError {
  if self.closed || self.cleanup_phase != 0 {
    raise SessionFailed(Invalid(Lifecycle))
  }
  self.require_owner()
  if self.terminal_cut != 0 || self.string_cut != 0 {
    return
  }
  if self.deferred_cut == 1 {
    return
  }
  if self.event_kind >= 3 {
    return
  }
  if self.event_ready {
    if self.event_kind != 1 && self.event_kind != 2 {
      raise SessionFailed(Invalid(EventCredit))
    }
    if self.deferred_cut == 0 {
      self.deferred_cut = 1
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
pub fn OnlineSession::check_deadline(
  self : OnlineSession,
) -> Unit raise OnlineSessionError {
  if self.closed || self.cleanup_phase != 0 {
    raise SessionFailed(Invalid(Lifecycle))
  }
  self.require_owner()
  if self.terminal_cut != 0 || self.string_cut != 0 {
    return
  }
  if self.deferred_cut == 1 {
    return
  }
  if self.event_kind >= 3 {
    return
  }
  if self.event_ready {
    // Poll only: never sample time or latch deadline intent while a frame is
    // pinned. Automatic expiry runs from the next credit-free progress call.
    return
  }
  self.begin_deadline_cut(self.lease_owner())
}
