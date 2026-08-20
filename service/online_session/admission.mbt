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
fn LunaOnlineInstance::claim_owner(
  self : LunaOnlineInstance,
) -> @request_admission.LunaPreparedRequestClaim {
  match self.claim {
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
    self.claim is None ||
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
/// Claims one off-reactor-prepared request into an idle persistent instance.
/// Busy, draining, and epoch-exhaustion disposition is decided before the
/// prepared owner can be consumed. Accepted credit is prepared before the
/// one-shot scheduler-request claim and retired if lower admission rejects.
pub fn LunaOnlineInstance::begin(
  self : LunaOnlineInstance,
  prepared : @request_admission.LunaPreparedRequest,
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
  let stream_preference = prepared.stream_preference() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
  }
  if stream_preference != Streaming {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineStreaming))
  }
  let model_identity = prepared.model_identity() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
  }
  if model_identity != self.expected_model {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineModelIdentity))
  }
  let tokenizer_digest = prepared.tokenizer_digest() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
  }
  if tokenizer_digest != self.expected_tokenizer {
    raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
  }
  let inference_limits = prepared.inference_limits() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
  }
  if inference_limits != self.inference {
    raise LunaOnlineInstanceFailed(
      LunaOnlineInvalid(LunaOnlineInferenceEnvelope),
    )
  }
  let effective = prepared.effective_limits() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
  }
  let input_tokens = prepared.input_token_count() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
  }
  self.writer.prepare_accepted(model_identity, effective) catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineWireFailure)
  }
  let claim = prepared.take_claim() catch {
    _ => {
      try! self.writer.retire()
      raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
    }
  }
  self.lease_owner().admit(claim.scheduler_request()) catch {
    _ => {
      try! self.writer.retire()
      raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
    }
  }
  let next_epoch = self.request_epoch + 1UL
  self.claim = Some(claim)
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
