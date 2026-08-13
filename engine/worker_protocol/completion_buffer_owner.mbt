///|
fn CompletionBuffer::require_state(
  self : CompletionBuffer,
  expected : CompletionBufferState,
) -> Unit raise WorkerProtocolError {
  if self.state != expected {
    raise InvalidCompletionBufferState(expected~, actual=self.state)
  }
}

///|
fn CompletionBuffer::entry_index_for_slot(
  self : CompletionBuffer,
  slot : CompletionSlot,
) -> Int raise WorkerProtocolError {
  let value = slot.value()
  if value < 0 || value >= self.slot_to_entry.length() {
    raise InvalidCompletionSlot
  }
  self.slot_to_entry[value]
}

///|
fn CompletionBuffer::clear_slot_index(self : CompletionBuffer) -> Unit {
  for index in 0..<self.entry_count {
    self.slot_to_entry[self.slots[index].value()] = -1
  }
}

///|
fn CompletionBuffer::append(
  self : CompletionBuffer,
  slot : CompletionSlot,
  request_id : @inference.RequestId,
  request_generation : @inference.RequestGeneration,
  kind : CompletionEntryKind,
  processed_tokens : Int,
  token_id : Int,
  failure : WorkerFailure,
) -> Unit raise WorkerProtocolError {
  self.require_state(CompletionWritable)
  if self.entry_count >= self.limits.max_plan_rows() {
    raise CompletionCountMismatch
  }
  let previous = self.entry_index_for_slot(slot)
  if previous != -1 {
    raise DuplicateCompletionSlot(index=self.entry_count)
  }
  let index = self.entry_count
  self.slots[index] = slot
  self.request_ids[index] = request_id
  self.request_generations[index] = request_generation
  self.kinds[index] = kind
  self.processed_token_counts[index] = processed_tokens
  self.token_ids[index] = token_id
  self.failures[index] = failure
  self.slot_to_entry[slot.value()] = index
  self.entry_count = index + 1
}

///|
fn CompletionBuffer::require_direct_access(
  self : CompletionBuffer,
) -> Unit raise WorkerProtocolError {
  self.require_state(CompletionWritable)
  if self.writer_open {
    raise CompletionWriterAlreadyOpen
  }
}

///|
pub fn CompletionBuffer::new(
  limits : WorkerProtocolLimits,
) -> CompletionBuffer raise WorkerProtocolError {
  let request_id = try! @inference.RequestId::new(1UL)
  let generation = @inference.RequestGeneration::initial()
  let slot = CompletionSlot::new(0, limits)
  {
    limits,
    state: CompletionWritable,
    epoch: 1UL,
    sequence_value: 1UL,
    model_plan_generation_value: 1UL,
    entry_count: 0,
    writer_open: false,
    slot_to_entry: FixedArray::make(limits.max_completion_slots(), -1),
    slots: FixedArray::make(limits.max_plan_rows(), slot),
    request_ids: FixedArray::make(limits.max_plan_rows(), request_id),
    request_generations: FixedArray::make(limits.max_plan_rows(), generation),
    kinds: FixedArray::make(limits.max_plan_rows(), FailedCompletion),
    processed_token_counts: FixedArray::make(limits.max_plan_rows(), 0),
    token_ids: FixedArray::make(limits.max_plan_rows(), 0),
    failures: FixedArray::make(limits.max_plan_rows(), CancelledBeforeExecution),
  }
}

///|
pub fn CompletionBuffer::append_prefill_completed(
  self : CompletionBuffer,
  slot~ : CompletionSlot,
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  processed_tokens~ : Int,
) -> Unit raise WorkerProtocolError {
  self.require_direct_access()
  self.append_prefill_completed_unleased(
    slot~,
    request_id~,
    request_generation~,
    processed_tokens~,
  )
}

///|
fn CompletionBuffer::append_prefill_completed_unleased(
  self : CompletionBuffer,
  slot~ : CompletionSlot,
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  processed_tokens~ : Int,
) -> Unit raise WorkerProtocolError {
  if processed_tokens <= 0 || processed_tokens > self.limits.max_plan_tokens() {
    raise CompletionTokenCountMismatch(index=self.entry_count)
  }
  self.append(
    slot,
    request_id,
    request_generation,
    PrefillCompletion,
    processed_tokens,
    0,
    CancelledBeforeExecution,
  )
}

///|
/// Appends the final prompt completion and its first sampled output token.
pub fn CompletionBuffer::append_final_prefill_completed(
  self : CompletionBuffer,
  slot~ : CompletionSlot,
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  processed_tokens~ : Int,
  token_id~ : Int,
) -> Unit raise WorkerProtocolError {
  self.require_direct_access()
  self.append_final_prefill_completed_unleased(
    slot~,
    request_id~,
    request_generation~,
    processed_tokens~,
    token_id~,
  )
}

///|
fn CompletionBuffer::append_final_prefill_completed_unleased(
  self : CompletionBuffer,
  slot~ : CompletionSlot,
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  processed_tokens~ : Int,
  token_id~ : Int,
) -> Unit raise WorkerProtocolError {
  if processed_tokens <= 0 || processed_tokens > self.limits.max_plan_tokens() {
    raise CompletionTokenCountMismatch(index=self.entry_count)
  }
  if token_id < 0 || token_id > self.limits.max_token_id() {
    raise InvalidCompletionToken(slot=slot.value())
  }
  self.append(
    slot,
    request_id,
    request_generation,
    FinalPrefillCompletion,
    processed_tokens,
    token_id,
    CancelledBeforeExecution,
  )
}

///|
pub fn CompletionBuffer::append_decode_completed(
  self : CompletionBuffer,
  slot~ : CompletionSlot,
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  token_id~ : Int,
) -> Unit raise WorkerProtocolError {
  self.require_direct_access()
  self.append_decode_completed_unleased(
    slot~,
    request_id~,
    request_generation~,
    token_id~,
  )
}

///|
fn CompletionBuffer::append_decode_completed_unleased(
  self : CompletionBuffer,
  slot~ : CompletionSlot,
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  token_id~ : Int,
) -> Unit raise WorkerProtocolError {
  if token_id < 0 || token_id > self.limits.max_token_id() {
    raise InvalidCompletionToken(slot=slot.value())
  }
  self.append(
    slot,
    request_id,
    request_generation,
    DecodeCompletion,
    0,
    token_id,
    CancelledBeforeExecution,
  )
}

///|
pub fn CompletionBuffer::append_failed(
  self : CompletionBuffer,
  slot~ : CompletionSlot,
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  failure~ : WorkerFailure,
) -> Unit raise WorkerProtocolError {
  self.require_direct_access()
  self.append_failed_unleased(slot~, request_id~, request_generation~, failure~)
}

///|
fn CompletionBuffer::append_failed_unleased(
  self : CompletionBuffer,
  slot~ : CompletionSlot,
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  failure~ : WorkerFailure,
) -> Unit raise WorkerProtocolError {
  self.append(
    slot,
    request_id,
    request_generation,
    FailedCompletion,
    0,
    0,
    failure,
  )
}

///|
/// Freezes this ingress for the exact authenticated submitted plan without
/// copying entries. Exact row identity is checked by `validate_for` before the
/// scheduler consumes or retires the associated plan.
pub fn CompletionBuffer::submit_for(
  self : CompletionBuffer,
  plan : SubmittedSchedulePlan,
) -> SubmittedCompletion raise WorkerProtocolError {
  self.require_direct_access()
  self.submit_for_unleased(plan)
}

///|
fn CompletionBuffer::submit_for_unleased(
  self : CompletionBuffer,
  plan : SubmittedSchedulePlan,
) -> SubmittedCompletion raise WorkerProtocolError {
  self.require_state(CompletionWritable)
  if self.entry_count == 0 {
    raise CompletionCountMismatch
  }
  self.validate_against(plan)
  self.sequence_value = plan.sequence_wire()
  self.model_plan_generation_value = plan.model_plan_generation_wire()
  self.state = CompletionSubmitted
  SubmittedCompletion(self, self.epoch)
}

///|
/// Nonmutating scheduler preflight for the exact completion owner, completion
/// epoch, plan owner, plan epoch, plan sequence, model generation, and rows.
/// Callers may validate and read all entries before committing scheduler state;
/// `consume` repeats this authentication before changing protocol lifecycle.
pub fn CompletionBuffer::validate_submitted(
  self : CompletionBuffer,
  submitted : SubmittedCompletion,
  plan : SubmittedSchedulePlan,
) -> Unit raise WorkerProtocolError {
  submitted.authenticate_owner(self)
  submitted.validate_for(plan)
}

///|
/// Returns this owner's exact submitted completion after reauthenticating it
/// against the supplied still-live plan. This supports retry after a caller's
/// publication preflight rejects without reopening or copying the completion.
pub fn CompletionBuffer::current_submitted(
  self : CompletionBuffer,
  plan : SubmittedSchedulePlan,
) -> SubmittedCompletion raise WorkerProtocolError {
  self.require_state(CompletionSubmitted)
  let submitted = SubmittedCompletion(self, self.epoch)
  submitted.validate_for(plan)
  submitted
}

///|
pub fn CompletionBuffer::consume(
  self : CompletionBuffer,
  submitted : SubmittedCompletion,
  plan : SubmittedSchedulePlan,
) -> ConsumedCompletion raise WorkerProtocolError {
  submitted.authenticate_owner(self)
  submitted.validate_for(plan)
  self.clear_slot_index()
  self.state = Consumed
  ConsumedCompletion(self, self.epoch)
}

///|
fn ConsumedCompletion::authenticate_owner(
  self : ConsumedCompletion,
  owner : CompletionBuffer,
) -> Unit raise WorkerProtocolError {
  if !physical_equal(self.0, owner) {
    raise CompletionBufferMismatch
  }
  if self.1 != owner.epoch {
    raise StaleCompletionBufferEpoch
  }
  owner.require_state(Consumed)
}

///|
pub fn CompletionBuffer::reset(
  self : CompletionBuffer,
  consumed : ConsumedCompletion,
) -> Unit raise WorkerProtocolError {
  consumed.authenticate_owner(self)
  if self.epoch == 0xffffffffffffffffUL {
    raise IdentityOverflow(CompletionBufferEpochIdentity)
  }
  self.clear_slot_index()
  self.entry_count = 0
  self.writer_open = false
  self.epoch = self.epoch + 1UL
  self.state = CompletionWritable
}

///|
pub fn CompletionBuffer::state(
  self : CompletionBuffer,
) -> CompletionBufferState {
  self.state
}

///|
pub fn CompletionBuffer::epoch(self : CompletionBuffer) -> UInt64 {
  self.epoch
}
