///|
pub fn OnlineSessionPreparation::state(
  self : OnlineSessionPreparation,
) -> OnlineSessionPreparationState {
  self.state
}

///|
pub fn OnlineSessionPreparation::take_ready(
  self : OnlineSessionPreparation,
) -> OnlineSession raise OnlineSessionPreparationError {
  if self.consumed || self.state != OnlineSessionReady {
    raise preparation_failure(Invalid(Lifecycle))
  }
  self.consumed = true
  match self.session {
    Some(session) => {
      self.session = None
      session
    }
    None => abort("online session ready invariant")
  }
}

///|
pub fn OnlineSessionPreparation::take_cleanup(
  self : OnlineSessionPreparation,
) -> FailedOnlineSessionPreparation raise OnlineSessionPreparationError {
  if self.consumed || self.state != OnlineSessionCleanupRequired {
    raise preparation_failure(Invalid(Lifecycle))
  }
  self.consumed = true
  self.session = None
  self.cleanup
}

///|
pub fn FailedOnlineSessionPreparation::cause(
  self : FailedOnlineSessionPreparation,
) -> OnlineSessionFailure {
  match self.cause_kind {
    1 => ServiceFailure
    2 => {
      if self.service_cleanup is None {
        abort("online cleanup preparation cause invariant")
      }
      ServicePreparationFailure
    }
    3 => AdmissionFailure
    _ => Invalid(Ownership)
  }
}

///|
pub fn FailedOnlineSessionPreparation::is_closed(
  self : FailedOnlineSessionPreparation,
) -> Bool {
  self.closed
}

///|
pub fn FailedOnlineSessionPreparation::retry_close(
  self : FailedOnlineSessionPreparation,
) -> Unit raise OnlineSessionPreparationError {
  if self.closed {
    return
  }
  match self.lease {
    Some(lease) => {
      lease.retry_close_empty() catch {
        _ => raise preparation_failure(ServiceFailure)
      }
      self.lease = None
      self.closed = true
    }
    None =>
      match self.service_cleanup {
        Some(cleanup) => {
          cleanup.retry_close() catch {
            _ => raise preparation_failure(ServicePreparationFailure)
          }
          if cleanup.is_closed() {
            self.service_cleanup = None
            self.closed = true
          }
        }
        None => abort("online session cleanup invariant")
      }
  }
}

///|
fn OnlineSession::require_owner(self : OnlineSession) -> Unit {
  if self.lease is None ||
    self.output.length() == 0 ||
    self.max_new_tokens <= 0 ||
    self.input_tokens != self.admitted.input_token_count() {
    abort("online session owner invariant")
  }
}

///|
pub fn OnlineSession::has_event(self : OnlineSession) -> Bool {
  if self.closed || self.cleanup_phase != 0 {
    return false
  }
  self.require_owner()
  self.event_ready
}

///|
pub fn OnlineSession::event_length(
  self : OnlineSession,
) -> Int raise OnlineSessionError {
  if self.closed || self.cleanup_phase != 0 {
    raise SessionFailed(Invalid(EventCredit))
  }
  self.require_owner()
  if !self.event_ready {
    raise SessionFailed(Invalid(EventCredit))
  }
  self.writer.length() catch {
    _ => raise SessionFailed(WireFailure)
  }
}

///|
pub fn OnlineSession::copy_event_to(
  self : OnlineSession,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Int raise OnlineSessionError {
  if self.closed || self.cleanup_phase != 0 {
    raise SessionFailed(Invalid(EventCredit))
  }
  self.require_owner()
  if !self.event_ready {
    raise SessionFailed(Invalid(EventCredit))
  }
  self.writer.copy_to(destination, destination_offset~) catch {
    _ => raise SessionFailed(WireFailure)
  }
}

///|
pub fn OnlineSession::ack_event(
  self : OnlineSession,
) -> Unit raise OnlineSessionError {
  if self.closed || self.cleanup_phase != 0 {
    raise SessionFailed(Invalid(EventCredit))
  }
  self.require_owner()
  if !self.event_ready {
    raise SessionFailed(Invalid(EventCredit))
  }
  match self.event_kind {
    1 => {
      if self.accepted_acked {
        raise SessionFailed(Invalid(EventCredit))
      }
      self.writer.retire() catch {
        _ => raise SessionFailed(WireFailure)
      }
      self.event_ready = false
      self.accepted_acked = true
    }
    2 => {
      self.writer.retire() catch {
        _ => raise SessionFailed(WireFailure)
      }
      self.event_ready = false
    }
    3 => {
      self.writer.retire() catch {
        _ => raise SessionFailed(WireFailure)
      }
      try! self.writer.prepare_completed(
        if self.finish_kind == 1 {
          StopSequence
        } else if self.finish_kind == 2 {
          TokenLimit
        } else {
          Cancelled
        },
        input_tokens=self.input_tokens,
        cached_input_tokens=0,
        output_tokens=self.output_tokens,
        self.output,
        tail_offset=0,
        tail_length=self.final_tail_length,
      )
      self.event_kind = 4
      self.event_ready = true
    }
    4 => {
      self.writer.retire() catch {
        _ => raise SessionFailed(WireFailure)
      }
      self.event_ready = false
      self.cleanup_phase = 2
    }
    5 => {
      self.writer.retire() catch {
        _ => raise SessionFailed(WireFailure)
      }
      try! self.writer.prepare_failed(
        self.deadline_code,
        code_offset=0,
        code_length=self.deadline_code.length(),
        retryable=false,
      )
      self.event_kind = 6
      self.event_ready = true
    }
    6 => {
      self.writer.retire() catch {
        _ => raise SessionFailed(WireFailure)
      }
      self.event_ready = false
      self.cleanup_phase = 2
    }
    _ => abort("online session event kind invariant")
  }
}

///|
/// Begins the exact cancellation cut once. Cleanup remains owner-resident and
/// is advanced through `progress_cleanup`; this call never assumes that the
/// terminal publication is already visible.
pub fn OnlineSession::begin_abort(
  self : OnlineSession,
) -> Unit raise OnlineSessionError {
  if self.closed {
    return
  }
  self.require_owner()
  if self.cleanup_phase != 0 {
    return
  }
  if self.event_ready {
    self.writer.retire() catch {
      _ => raise SessionFailed(WireFailure)
    }
    self.event_ready = false
    self.accepted_acked = true
  }
  if self.event_kind >= 3 && self.event_kind <= 6 {
    self.cleanup_phase = 2
    return
  }
  if self.string_cut != 0 || self.terminal_cut != 0 {
    self.cleanup_phase = 1
    return
  }
  self.deferred_cut = 0
  let lease = match self.lease {
    Some(value) => value
    None => abort("online session lease invariant")
  }
  let reservation = lease.reserve_cancel() catch {
    _ => raise SessionFailed(ServiceFailure)
  }
  match reservation {
    OnlineCancelReserved => {
      let committed = lease.commit_cancel() catch {
        _ => raise SessionFailed(ServiceFailure)
      }
      match committed {
        OnlineCancelCommitted => ()
        OnlineCancelReservationAbsent => raise SessionFailed(Invalid(Lifecycle))
      }
    }
    OnlineCancelNaturalTerminal => {
      self.cleanup_phase = 1
      return
    }
  }
  self.cleanup_phase = 1
}

///|
/// Advances at most one cleanup transition: one pending worker operation, one
/// suppressed/terminal publication, or one close attempt. The exact lease and
/// cleanup authority remain retained after every reported failure.
pub fn OnlineSession::progress_cleanup(
  self : OnlineSession,
) -> OnlineSessionCleanupProgress raise OnlineSessionError {
  if self.closed {
    return OnlineSessionCleanupClosed
  }
  self.require_owner()
  if self.cleanup_phase == 0 {
    raise SessionFailed(Invalid(Lifecycle))
  }
  let lease = match self.lease {
    Some(value) => value
    None => abort("online session lease invariant")
  }
  if self.cleanup_phase == 1 {
    let available = lease.has_publication() catch {
      _ => raise SessionFailed(ServiceFailure)
    }
    if !available {
      try lease.progress() catch {
        _ => {
          self.cleanup_phase = 5
          self.recovery_close = true
        }
      } noraise {
        _ => ()
      }
      return OnlineSessionCleanupPending
    }
    let publication = lease.take_publication() catch {
      _ => raise SessionFailed(ServiceFailure)
    }
    match publication.kind() {
      OnlineTerminalPublication => self.cleanup_phase = 2
      OnlineSuppressedTokenPublication | OnlineGeneratedTokenPublication => ()
    }
    return OnlineSessionCleanupPending
  }
  if self.cleanup_phase == 2 {
    if self.recovery_close {
      lease.close_terminal() catch {
        _ => {
          self.cleanup_phase = 3
          raise SessionFailed(ServiceFailure)
        }
      }
    } else {
      lease.shutdown_clean() catch {
        _ => {
          self.cleanup_phase = 3
          raise SessionFailed(ServiceFailure)
        }
      }
    }
  } else if self.cleanup_phase == 3 {
    lease.retry_close_terminal() catch {
      _ => raise SessionFailed(ServiceFailure)
    }
  } else if self.cleanup_phase == 5 {
    let recovery = lease.progress_terminal_recovery(ResourceUnavailable) catch {
      _ => raise SessionFailed(ServiceFailure)
    }
    match recovery {
      OnlineTerminalRecoveryPending => ()
      OnlineTerminalRecoveryPublicationReady => self.cleanup_phase = 1
    }
    return OnlineSessionCleanupPending
  } else {
    abort("online session cleanup phase invariant")
  }
  self.lease = None
  self.closed = true
  self.cleanup_phase = 4
  OnlineSessionCleanupClosed
}
