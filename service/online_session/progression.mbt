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
fn OnlineSession::enter_output_cleanup(
  self : OnlineSession,
  lease : @worker_service.OnlineWorkerLease,
  natural_terminal : Bool,
) -> Unit {
  if natural_terminal {
    self.cleanup_phase = 1
    return
  }
  let committed = try! lease.commit_cancel()
  match committed {
    OnlineCancelCommitted => self.cleanup_phase = 1
    OnlineCancelReservationAbsent => abort("online output cut invariant")
  }
}

///|
fn OnlineSession::decode_token(
  self : OnlineSession,
  lease : @worker_service.OnlineWorkerLease,
  publication : @worker_service.OnlineWorkerPublication,
  natural_terminal : Bool,
) -> Unit raise OnlineSessionError {
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
    self.enter_output_cleanup(lease, natural_terminal)
    raise SessionFailed(OutputFailure)
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
      self.cleanup_phase = 1
    } else {
      self.enter_output_cleanup(lease, natural_terminal)
    }
    raise SessionFailed(OutputFailure)
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
    self.cleanup_phase = 2
    raise SessionFailed(OutputFailure)
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
  if self.string_cut != 0 {
    self.progress_string_cut(lease)
    if self.event_ready {
      return OnlineSessionEventReady
    }
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
    return OnlineSessionPending
  }
  try lease.progress() catch {
    _ => {
      self.cleanup_phase = 5
      self.recovery_close = true
      raise SessionFailed(ServiceFailure)
    }
  } noraise {
    _ => ()
  }
  OnlineSessionPending
}
