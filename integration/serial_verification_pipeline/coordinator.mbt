///|
/// Draft inputs include the already-emitted seed in row zero. A completed
/// verification plan stays immutable in this owner's storage through replay.
pub fn TwoRankGreedyVerification::begin(
  self : Self,
  plan : @wire.ValidatedPlanFrame,
  epoch~ : UInt64,
  remaining_new_tokens~ : Int,
  local_now_millis~ : UInt64,
) -> Unit raise {
  guard !self.busy &&
    self.phase is (Ready | Committed) &&
    epoch > self.epoch &&
    epoch < 0xffffffffffffffffUL &&
    remaining_new_tokens > 0 &&
    local_now_millis >= self.tick &&
    local_now_millis <= 0xffffffffffffffffUL - self.timeout &&
    plan.prefill_count() == 1 &&
    plan.decode_count() == 0 &&
    plan.token_count() <= self.inputs.length() else {
    raise @control.StageControlError::InvalidState
  }
  let row = plan.prefill_row(0)
  guard row.produces_token() &&
    row.sampling_mode_value() == 0 &&
    row.token_start() == 0 &&
    row.token_length() == plan.token_count() &&
    plan.sequence_value() < 0xffffffffffffffffUL else {
    raise @control.StageControlError::InvalidMessage
  }
  self.phase = Failed
  self.live = plan.token_count()
  for i in 0..<self.live {
    self.inputs[i] = plan.token_at(i)
  }
  plan.copy_to(self.storage, destination_offset=0)
  let retained = self.original.load(self.storage, plan.length())
  self.ingress.begin(retained, self.completions.writer_for(retained))
  self.remote.prepare(retained, epoch~, local_now_millis~, tentative=true)
  self.epoch = epoch
  self.tick = local_now_millis
  self.deadline = local_now_millis + self.timeout
  self.remaining = remaining_new_tokens
  self.local_ready = false
  self.remote_ready = false
  self.activation_done = false
  self.save_begun = false
  self.save_done = false
  self.restore_done = false
  self.replaying = false
  self.failed = false
  self.decision = None
  self.phase = Preparing
}

///|
/// Only committed target tokens may escape. A partial prefix remains private
/// while either rollback or accepted-prefix replay is outstanding.
pub fn TwoRankGreedyVerification::result(
  self : Self,
) -> @greedy.VerifiedGreedyPrefix raise {
  guard self.phase is Committed && self.decision is Some(decision) else {
    raise @control.StageControlError::InvalidState
  }
  decision
}

///|
pub fn TwoRankGreedyVerification::token_at(self : Self, row : Int) -> Int raise {
  let decision = self.result()
  guard row >= 0 && row < decision.emitted_token_count() else {
    raise @control.StageControlError::InvalidState
  }
  self.outputs[row]
}

///|
/// Rejection replay consumes an additional transport epoch. Callers must start
/// their next transaction strictly after this value, not a guessed token count.
pub fn TwoRankGreedyVerification::last_epoch(self : Self) -> UInt64 {
  self.epoch
}

///|
pub fn TwoRankGreedyVerification::is_failed(self : Self) -> Bool {
  self.phase is Failed
}

///|
/// Release the completed request on both ranks. The previous receipt becomes
/// unreadable immediately; reuse starts only after the remote release retires.
pub fn TwoRankGreedyVerification::release_request(
  self : Self,
  local_now_millis~ : UInt64,
) -> Unit raise {
  guard !self.busy &&
    self.phase is Committed &&
    local_now_millis >= self.tick &&
    local_now_millis <= 0xffffffffffffffffUL - self.timeout else {
    raise @control.StageControlError::InvalidState
  }
  self.phase = Failed
  self.ingress.release_request()
  self.remote.release_request(local_now_millis~)
  self.tick = local_now_millis
  self.deadline = local_now_millis + self.timeout
  self.decision = None
  self.phase = Releasing
}

///|
pub fn TwoRankGreedyVerification::close(self : Self) -> Unit raise {
  guard !self.busy else { raise @control.StageControlError::InvalidState }
  if self.phase is Closed {
    return
  }
  self.phase = Failed
  self.remote.close()
  (self.close_activation)()
  // This effect drains model work before resolving abandoned device backups.
  // It must run before ingress.close() can release the leased state allocations.
  (self.close_local_state)()
  self.ingress.close()
  self.phase = Closed
}
