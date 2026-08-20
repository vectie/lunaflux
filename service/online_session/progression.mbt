///|
fn LunaOnlineInstance::begin_output_failure(
  self : LunaOnlineInstance,
  lease : @worker_service.OnlineWorkerLease,
  natural_terminal : Bool,
) -> Unit {
  self.failure_kind = OutputLatchedFailure
  if natural_terminal {
    self.terminal_cut = NaturalOutputFailureCut
    return
  }
  let committed = try! lease.commit_cancel()
  match committed {
    OnlineCancelCommitted => self.terminal_cut = OutputCancellationCut
    OnlineCancelReservationAbsent => abort("online output cut invariant")
  }
}

///|
fn LunaOnlineInstance::decode_token(
  self : LunaOnlineInstance,
  lease : @worker_service.OnlineWorkerLease,
  publication : @worker_service.OnlineWorkerPublication,
  natural_terminal : Bool,
) -> Unit {
  let token_id = try! publication.token_id()
  let claim = self.claim_owner()
  if claim.is_stop_token(token_id) {
    abort("online natural stop token decode invariant")
  }
  self.output_tokens += 1
  let step = claim.push_token_into_status(
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
      self.string_cut = NaturalStringCut
    } else {
      let committed = try! lease.commit_cancel()
      match committed {
        OnlineCancelCommitted => self.string_cut = CancelledStringCut
        OnlineCancelReservationAbsent => abort("online string cut invariant")
      }
    }
    if step.copied_bytes() == 0 {
      return
    }
  }
  let published = self.events.publish_token_status(
    token_id~,
    position=try! publication.position(),
    self.output,
    source_offset=0,
    length=step.copied_bytes(),
  )
  match published {
    LunaEventPublishReady => ()
    LunaEventPublishPayloadInvalid => {
      if step.stop_matched() {
        self.failure_kind = OutputLatchedFailure
        self.terminal_cut = if natural_terminal {
          NaturalOutputFailureCut
        } else {
          OutputCancellationCut
        }
        self.string_cut = NoStringCut
      } else {
        self.begin_output_failure(lease, natural_terminal)
      }
      return
    }
    LunaEventPublishEpochExhausted | LunaEventPublishCreditUnavailable =>
      abort("luna semantic event publication invariant")
  }
  if !step.stop_matched() && !natural_terminal {
    try! lease.abort_cancel()
  }
  self.mark_event_published(TokenEvent)
}

///|
fn LunaOnlineInstance::prepare_string_stop_terminal(
  self : LunaOnlineInstance,
) -> Unit {
  self.string_cut = NoStringCut
  self.finish_kind = FinishStopSequence
  self.final_tail_length = 0
  try! self.events.publish_usage(
    input_tokens=self.input_tokens,
    cached_input_tokens=0,
    output_tokens=self.output_tokens,
  )
  self.mark_event_published(CompletionUsageEvent)
}

///|
fn LunaOnlineInstance::progress_string_cut(
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
    OnlineTerminalPublication => {
      let reason = try! publication.terminal_reason()
      if self.string_cut == CancelledStringCut {
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
      if self.string_cut != CancelledStringCut {
        abort("online string suppression invariant")
      }
      self.output_tokens += 1
    }
    OnlineGeneratedTokenPublication =>
      abort("online natural string terminal adjacency invariant")
  }
}

///|
fn LunaOnlineInstance::prepare_natural_terminal(
  self : LunaOnlineInstance,
  reason : @worker_service.OnlineWorkerTerminalReason,
) -> Unit raise LunaOnlineInstanceError {
  if reason == OnlineWorkerFailed {
    self.failure_kind = WorkerLatchedFailure
    self.recovery_close = true
    self.final_tail_length = 0
    try! self.events.publish_usage(
      input_tokens=self.input_tokens,
      cached_input_tokens=0,
      output_tokens=self.output_tokens,
    )
    self.mark_event_published(FailureUsageEvent)
    return
  }
  self.finish_kind = match reason {
    OnlineStopTokenReached => FinishStopSequence
    OnlineMaximumOutputReached => FinishTokenLimit
    _ => {
      self.recovery_close = true
      self.phase = LunaInstanceClosingTerminalPhase
      raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
    }
  }
  let tail = self
    .claim_owner()
    .finish_into_status(self.output, destination_offset=0)
  if !tail.is_ready() {
    self.failure_kind = OutputLatchedFailure
    self.final_tail_length = 0
    try! self.events.publish_usage(
      input_tokens=self.input_tokens,
      cached_input_tokens=0,
      output_tokens=self.output_tokens,
    )
    self.mark_event_published(FailureUsageEvent)
    return
  }
  self.final_tail_length = tail.copied_bytes()
  try! self.events.publish_usage(
    input_tokens=self.input_tokens,
    cached_input_tokens=0,
    output_tokens=self.output_tokens,
  )
  self.mark_event_published(CompletionUsageEvent)
}

///|
fn LunaOnlineInstance::consume_natural_publication(
  self : LunaOnlineInstance,
  lease : @worker_service.OnlineWorkerLease,
) -> Unit raise LunaOnlineInstanceError {
  let publication = try! lease.take_publication()
  match publication.kind() {
    OnlineGeneratedTokenPublication => {
      let token_id = try! publication.token_id()
      if self.claim_owner().is_stop_token(token_id) {
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
pub fn LunaOnlineInstance::progress(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
) -> LunaOnlineInstanceProgress raise LunaOnlineInstanceError {
  self.require_ticket(ticket)
  if self.phase != LunaRequestActivePhase {
    return LunaOnlineInstanceRequestRetirementRequired
  }
  if self.events.has_event() {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit))
  }
  if !self.accepted_acked {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineLifecycle))
  }
  self.require_owner()
  let lease = self.lease_owner()
  if self.recovery_close {
    return LunaOnlineInstanceTerminalizationRequired
  }
  if self.terminal_cut != NoTerminalCut {
    match self.terminal_cut {
      OutputCancellationCut | WorkerFailureCut =>
        return LunaOnlineInstanceTerminalizationRequired
      _ => ()
    }
    self.progress_terminal_cut(lease)
    if self.recovery_close {
      return LunaOnlineInstanceTerminalizationRequired
    }
    if self.events.has_event() {
      return LunaOnlineInstanceEventReady
    }
    return LunaOnlineInstancePending
  }
  if self.string_cut != NoStringCut {
    self.progress_string_cut(lease)
    if self.recovery_close {
      return LunaOnlineInstanceTerminalizationRequired
    }
    if self.events.has_event() {
      return LunaOnlineInstanceEventReady
    }
    return LunaOnlineInstancePending
  }
  if self.deferred_cancel {
    self.begin_caller_cut(lease)
    self.deferred_cancel = false
    if self.terminal_cut != NoTerminalCut {
      return LunaOnlineInstancePending
    }
  }
  if self.enforce_deadline(lease) {
    return LunaOnlineInstancePending
  }
  let available = try! lease.has_publication()
  if available {
    let reservation = try lease.reserve_cancel() catch {
      _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
    } noraise {
      value => value
    }
    match reservation {
      OnlineCancelReserved => {
        let publication = lease.take_reserved_token() catch {
          _ => {
            try! lease.abort_cancel()
            raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
          }
        }
        self.decode_token(lease, publication, false)
      }
      OnlineCancelNaturalTerminal => self.consume_natural_publication(lease)
    }
    if self.events.has_event() {
      return LunaOnlineInstanceEventReady
    }
    match self.terminal_cut {
      OutputCancellationCut | WorkerFailureCut =>
        return LunaOnlineInstanceTerminalizationRequired
      _ => ()
    }
    return LunaOnlineInstancePending
  }
  let step = try! lease.progress_status()
  if self.latch_worker_step_failure(step) {
    return LunaOnlineInstanceTerminalizationRequired
  }
  LunaOnlineInstancePending
}
