///|
fn OnlineSession::lease_owner(
  self : OnlineSession,
) -> @worker_service.OnlineWorkerLease {
  match self.lease {
    Some(value) => value
    None => abort("online session lease invariant")
  }
}

///|
fn OnlineSession::begin_output_failure(
  self : OnlineSession,
  lease : @worker_service.OnlineWorkerLease,
  natural_terminal : Bool,
) -> Unit {
  self.failure_kind = 2
  if natural_terminal {
    self.terminal_cut = 5
    return
  }
  let committed = try! lease.commit_cancel()
  match committed {
    OnlineCancelCommitted => self.terminal_cut = 3
    OnlineCancelReservationAbsent => abort("online output cut invariant")
  }
}

///|
fn OnlineSession::decode_token(
  self : OnlineSession,
  lease : @worker_service.OnlineWorkerLease,
  publication : @worker_service.OnlineWorkerPublication,
  natural_terminal : Bool,
) -> Unit {
  let token_id = try! publication.token_id()
  if self.admitted.is_stop_token(token_id) {
    abort("online natural stop token decode invariant")
  }
  self.output_tokens += 1
  let step = self.admitted.push_token_into_status(
    token_id,
    self.output,
    destination_offset=0,
  )
  if !step.is_ready() {
    self.begin_output_failure(lease, natural_terminal)
    return
  }
  if step.stop_matched() {
    if natural_terminal {
      self.string_cut = 2
    } else {
      let committed = try! lease.commit_cancel()
      match committed {
        OnlineCancelCommitted => self.string_cut = 1
        OnlineCancelReservationAbsent => abort("online string cut invariant")
      }
    }
    if step.copied_bytes() == 0 {
      return
    }
  }
  if !self.writer.prepare_token_bytes_status(
      token_id~,
      position=try! publication.position(),
      self.output,
      source_offset=0,
      length=step.copied_bytes(),
    ) {
    if step.stop_matched() {
      self.failure_kind = 2
      self.terminal_cut = if natural_terminal { 5 } else { 3 }
      self.string_cut = 0
    } else {
      self.begin_output_failure(lease, natural_terminal)
    }
    return
  }
  if !step.stop_matched() && !natural_terminal {
    try! lease.abort_cancel()
  }
  self.event_kind = 2
  self.event_ready = true
}

///|
fn OnlineSession::prepare_string_stop_terminal(self : OnlineSession) -> Unit {
  self.string_cut = 0
  self.finish_kind = 1
  self.final_tail_length = 0
  try! self.writer.prepare_usage(
    input_tokens=self.input_tokens,
    cached_input_tokens=0,
    output_tokens=self.output_tokens,
  )
  self.event_kind = 3
  self.event_ready = true
}

///|
fn OnlineSession::progress_string_cut(
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
    OnlineTerminalPublication => {
      let reason = try! publication.terminal_reason()
      if self.string_cut == 1 {
        if reason != OnlineCancelledByCaller {
          abort("online string cancellation terminal invariant")
        }
      } else if reason != OnlineMaximumOutputReached &&
        reason != OnlineStopTokenReached {
        abort("online natural string terminal invariant")
      }
      self.prepare_string_stop_terminal()
    }
    OnlineSuppressedTokenPublication => {
      if self.string_cut != 1 {
        abort("online string suppression invariant")
      }
      self.output_tokens += 1
    }
    OnlineGeneratedTokenPublication =>
      abort("online natural string terminal adjacency invariant")
  }
}

///|
fn OnlineSession::prepare_natural_terminal(
  self : OnlineSession,
  reason : @worker_service.OnlineWorkerTerminalReason,
) -> Unit raise OnlineSessionError {
  if reason == OnlineWorkerFailed {
    self.failure_kind = 3
    self.final_tail_length = 0
    try! self.writer.prepare_usage(
      input_tokens=self.input_tokens,
      cached_input_tokens=0,
      output_tokens=self.output_tokens,
    )
    self.event_kind = 5
    self.event_ready = true
    return
  }
  self.finish_kind = match reason {
    OnlineStopTokenReached => 1
    OnlineMaximumOutputReached => 2
    _ => {
      self.cleanup_phase = 2
      raise SessionFailed(ServiceFailure)
    }
  }
  let tail = self.admitted.finish_into_status(self.output, destination_offset=0)
  if !tail.is_ready() {
    self.failure_kind = 2
    self.final_tail_length = 0
    try! self.writer.prepare_usage(
      input_tokens=self.input_tokens,
      cached_input_tokens=0,
      output_tokens=self.output_tokens,
    )
    self.event_kind = 5
    self.event_ready = true
    return
  }
  self.final_tail_length = tail.copied_bytes()
  try! self.writer.prepare_usage(
    input_tokens=self.input_tokens,
    cached_input_tokens=0,
    output_tokens=self.output_tokens,
  )
  self.event_kind = 3
  self.event_ready = true
}

///|
fn OnlineSession::consume_natural_publication(
  self : OnlineSession,
  lease : @worker_service.OnlineWorkerLease,
) -> Unit raise OnlineSessionError {
  let publication = try! lease.take_publication()
  match publication.kind() {
    OnlineGeneratedTokenPublication => {
      let token_id = try! publication.token_id()
      if self.admitted.is_stop_token(token_id) {
        self.output_tokens += 1
        let terminal_available = try! lease.has_publication()
        if !terminal_available {
          abort("online stop-token terminal adjacency invariant")
        }
        let terminal = try! lease.take_publication()
        let terminal_reason = try! terminal.terminal_reason()
        if terminal.kind() != OnlineTerminalPublication ||
          terminal_reason != OnlineStopTokenReached {
          abort("online stop-token terminal identity invariant")
        }
        self.prepare_natural_terminal(OnlineStopTokenReached)
      } else {
        self.decode_token(lease, publication, true)
      }
    }
    OnlineTerminalPublication =>
      self.prepare_natural_terminal(try! publication.terminal_reason())
    OnlineSuppressedTokenPublication =>
      abort("online natural publication suppression invariant")
  }
}

///|
/// Advances at most one worker transition or publishes one canonical event.
/// Any pinned event credit rejects before scheduler, decoder, or worker state
/// can change.
pub fn OnlineSession::progress(
  self : OnlineSession,
) -> OnlineSessionProgress raise OnlineSessionError {
  if self.event_ready {
    raise SessionFailed(Invalid(EventCredit))
  }
  if self.closed || self.cleanup_phase != 0 || !self.accepted_acked {
    raise SessionFailed(Invalid(Lifecycle))
  }
  self.require_owner()
  let lease = self.lease_owner()
  if self.recovery_close {
    return OnlineSessionTerminalizationRequired
  }
  if self.terminal_cut != 0 {
    if self.terminal_cut == 3 || self.terminal_cut == 4 {
      return OnlineSessionTerminalizationRequired
    }
    self.progress_terminal_cut(lease)
    if self.recovery_close {
      return OnlineSessionTerminalizationRequired
    }
    if self.event_ready {
      return OnlineSessionEventReady
    }
    return OnlineSessionPending
  }
  if self.string_cut != 0 {
    self.progress_string_cut(lease)
    if self.recovery_close {
      return OnlineSessionTerminalizationRequired
    }
    if self.event_ready {
      return OnlineSessionEventReady
    }
    return OnlineSessionPending
  }
  if self.deferred_cut != 0 {
    let deferred = self.deferred_cut
    if deferred == 1 {
      self.begin_caller_cut(lease)
    }
    self.deferred_cut = 0
    if self.terminal_cut != 0 {
      return OnlineSessionPending
    }
  }
  if self.enforce_deadline(lease) {
    return OnlineSessionPending
  }
  let available = try! lease.has_publication()
  if available {
    let reservation = try lease.reserve_cancel() catch {
      _ => raise SessionFailed(ServiceFailure)
    } noraise {
      value => value
    }
    match reservation {
      OnlineCancelReserved => {
        let publication = lease.take_reserved_token() catch {
          _ => {
            try! lease.abort_cancel()
            raise SessionFailed(ServiceFailure)
          }
        }
        self.decode_token(lease, publication, false)
      }
      OnlineCancelNaturalTerminal => self.consume_natural_publication(lease)
    }
    if self.event_ready {
      return OnlineSessionEventReady
    }
    if self.terminal_cut == 3 || self.terminal_cut == 4 {
      return OnlineSessionTerminalizationRequired
    }
    return OnlineSessionPending
  }
  let step = try! lease.progress_status()
  if self.latch_worker_step_failure(step) {
    return OnlineSessionTerminalizationRequired
  }
  OnlineSessionPending
}
