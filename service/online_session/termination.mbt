///|
fn deadline_failure_code() -> FixedArray[Byte] {
  let source = b"deadline_exceeded"
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
    OnlineDeadlineCommitted => self.terminal_cut = 2
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
    self.event_kind = 5
  }
  self.terminal_cut = 0
  self.event_ready = true
}

///|
fn OnlineSession::progress_terminal_cut(
  self : OnlineSession,
  lease : @worker_service.OnlineWorkerLease,
) -> Unit raise OnlineSessionError {
  if !(try! lease.has_publication()) {
    try lease.progress() catch {
      _ => {
        self.cleanup_phase = 5
        self.recovery_close = true
        raise SessionFailed(ServiceFailure)
      }
    } noraise {
      _ => ()
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
      } else if reason != OnlineDeadlineReached {
        abort("online deadline terminal invariant")
      }
      self.prepare_cut_terminal()
    }
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
