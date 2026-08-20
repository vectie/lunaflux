///|
fn LunaOnlineInstance::lease_owner(
  self : LunaOnlineInstance,
) -> @worker_service.OnlineWorkerLease {
  match self.lease {
    Some(value) => value
    None => abort("luna online instance lease invariant")
  }
}

///|
fn LunaOnlineInstance::admitted_owner(
  self : LunaOnlineInstance,
) -> @request_admission.AdmittedRequest {
  match self.admitted {
    Some(value) => value
    None => abort("luna online instance request invariant")
  }
}

///|
fn LunaOnlineInstance::require_ticket(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
) -> Unit raise LunaOnlineInstanceError {
  if ticket.0 == 0UL ||
    ticket.0 != self.request_epoch ||
    self.admitted is None ||
    self.phase == LunaInstanceIdlePhase ||
    self.phase == LunaInstanceDrainingPhase ||
    self.phase == LunaInstanceRetryingEmptyClosePhase ||
    self.phase == LunaInstanceClosedPhase {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineRequestEpoch))
  }
}

///|
fn LunaOnlineInstance::require_active_ticket(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
) -> Unit raise LunaOnlineInstanceError {
  self.require_ticket(ticket)
  if self.phase != LunaRequestActivePhase {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineLifecycle))
  }
}

///|
/// Admits one request into an idle persistent instance. Busy or draining
/// disposition is decided before the receipt can be consumed. Accepted credit
/// is prepared before scheduler admission and retired if admission rejects.
pub fn LunaOnlineInstance::begin(
  self : LunaOnlineInstance,
  received : @request_admission.ReceivedRequest,
) -> LunaOnlineInstanceAdmission raise LunaOnlineInstanceError {
  if self.drain_requested ||
    self.recovery_close ||
    self.phase == LunaInstanceRecoveringTerminalPhase ||
    self.phase == LunaInstanceClosingTerminalPhase ||
    self.phase == LunaInstanceRetryingTerminalClosePhase ||
    self.phase == LunaInstanceDrainingPhase ||
    self.phase == LunaInstanceRetryingEmptyClosePhase ||
    self.phase == LunaInstanceClosedPhase {
    return LunaOnlineInstanceDraining
  }
  if self.phase != LunaInstanceIdlePhase {
    return LunaOnlineInstanceBusy
  }
  if self.request_epoch == 0xffffffffffffffffUL {
    self.drain_requested = true
    self.phase = LunaInstanceDrainingPhase
    return LunaOnlineInstanceDraining
  }
  let admitted = @request_admission.admit(
    received,
    self.tokenizer,
    self.expected_tokenizer,
    self.expected_model,
    self.inference,
    self.clock,
  ) catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
  }
  let request = admitted.scheduler_request()
  if request.stream_preference() != Streaming {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineStreaming))
  }
  let input_tokens = admitted.input_token_count()
  self.writer.prepare_accepted(
    request.model_identity(),
    request.effective_limits(),
  ) catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineWireFailure)
  }
  self.lease_owner().admit(request) catch {
    _ => {
      try! self.writer.retire()
      raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
    }
  }
  let next_epoch = self.request_epoch + 1UL
  self.admitted = Some(admitted)
  self.input_tokens = input_tokens
  self.request_epoch = next_epoch
  self.event_ready = true
  self.event_kind = AcceptedEvent
  self.accepted_acked = false
  self.output_tokens = 0
  self.final_tail_length = 0
  self.finish_kind = FinishPending
  self.string_cut = NoStringCut
  self.terminal_cut = NoTerminalCut
  self.deferred_cancel = false
  self.failure_kind = NoLatchedFailure
  self.recovery_close = false
  self.phase = LunaRequestActivePhase
  LunaOnlineInstanceAdmitted(LunaOnlineRequestTicket(next_epoch))
}
