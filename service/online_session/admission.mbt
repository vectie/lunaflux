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
  if ticket.lane >= 0 {
    ignore(self.require_multi_ticket(ticket))
    return
  }
  if ticket.epoch == 0UL ||
    ticket.epoch != self.request_epoch ||
    self.claim is None ||
    self.phase == LunaInstanceIdlePhase ||
    self.phase == LunaInstanceDrainingPhase ||
    self.phase == LunaInstanceRetryingEmptyClosePhase ||
    self.phase == LunaInstanceClosedPhase {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineRequestEpoch))
  }
}

///|
fn luna_online_admission(
  kind : LunaOnlineInstanceAdmissionKind,
  request_epoch : UInt64,
) -> LunaOnlineInstanceAdmission {
  {
    disposition: match kind {
      LunaOnlineInstanceAdmitted => 0
      LunaOnlineInstanceBusy => 1
      LunaOnlineInstanceDraining => 2
    },
    lane: -1,
    request_epoch,
  }
}

///|
pub fn LunaOnlineInstanceAdmission::kind(
  self : LunaOnlineInstanceAdmission,
) -> LunaOnlineInstanceAdmissionKind {
  match self.disposition {
    0 => LunaOnlineInstanceAdmitted
    1 => LunaOnlineInstanceBusy
    2 => LunaOnlineInstanceDraining
    _ => abort("luna online admission disposition invariant")
  }
}

///|
pub fn LunaOnlineInstanceAdmission::ticket(
  self : LunaOnlineInstanceAdmission,
) -> LunaOnlineRequestTicket raise LunaOnlineInstanceError {
  if self.disposition != 0 || self.request_epoch == 0UL {
    raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
  }
  { lane: self.lane, epoch: self.request_epoch }
}

///|
fn LunaOnlineInstance::require_active_ticket(
  self : LunaOnlineInstance,
  ticket : LunaOnlineRequestTicket,
) -> Unit raise LunaOnlineInstanceError {
  if ticket.lane >= 0 {
    ignore(self.require_active_multi_ticket(ticket))
    return
  }
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
    return luna_online_admission(LunaOnlineInstanceDraining, 0UL)
  }
  if self.phase != LunaInstanceIdlePhase {
    return luna_online_admission(LunaOnlineInstanceBusy, 0UL)
  }
  if self.request_epoch == 0xffffffffffffffffUL {
    self.drain_requested = true
    self.phase = LunaInstanceDrainingPhase
    return luna_online_admission(LunaOnlineInstanceDraining, 0UL)
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
  if !self.events.has_epoch_capacity(effective.max_new_tokens() + 3) {
    self.drain_requested = true
    self.phase = LunaInstanceDrainingPhase
    return luna_online_admission(LunaOnlineInstanceDraining, 0UL)
  }
  let input_tokens = prepared.input_token_count() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
  }
  self.events.publish_accepted(model_identity, effective) catch {
    LunaEventInvalid(LunaEventEpoch) => {
      self.drain_requested = true
      self.phase = LunaInstanceDrainingPhase
      return luna_online_admission(LunaOnlineInstanceDraining, 0UL)
    }
    _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
  }
  let claim = prepared.take_claim() catch {
    _ => {
      try! self.events.discard()
      raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
    }
  }
  self.lease_owner().admit(claim.scheduler_request()) catch {
    _ => {
      // Rejected lower admission retains no scheduler alias. Revoke the
      // claim-owned request resources before publishing the rollback outcome.
      try! claim.release()
      try! self.events.discard()
      raise LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure)
    }
  }
  let next_epoch = self.request_epoch + 1UL
  self.claim = Some(claim)
  self.input_tokens = input_tokens
  self.cached_input_tokens = 0
  self.request_epoch = next_epoch
  self.mark_event_published(AcceptedEvent)
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
  luna_online_admission(LunaOnlineInstanceAdmitted, next_epoch)
}
