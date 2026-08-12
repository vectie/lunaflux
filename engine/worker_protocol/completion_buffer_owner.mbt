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
fn CompletionBuffer::has_slot(
  self : CompletionBuffer,
  slot : CompletionSlot,
) -> Bool {
  for index in 0..<self.entry_count {
    if self.slots[index] == slot {
      return true
    }
  }
  false
}

///|
fn CompletionBuffer::append(
  self : CompletionBuffer,
  slot : CompletionSlot,
  request_id : @inference.RequestId,
  request_generation : @inference.RequestGeneration,
  kind : CompletionEntryKind,
  value : Int,
  failure : WorkerFailure,
) -> Unit raise WorkerProtocolError {
  self.require_state(CompletionWritable)
  if self.entry_count >= self.limits.max_plan_rows() {
    raise CompletionCountMismatch
  }
  if slot.value() < 0 || slot.value() >= self.limits.max_completion_slots() {
    raise InvalidCompletionSlot
  }
  if self.has_slot(slot) {
    raise DuplicateCompletionSlot(index=self.entry_count)
  }
  let index = self.entry_count
  self.slots[index] = slot
  self.request_ids[index] = request_id
  self.request_generations[index] = request_generation
  self.kinds[index] = kind
  self.values[index] = value
  self.failures[index] = failure
  self.entry_count = index + 1
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
    sequence: PlanSequence::first(),
    model_plan_generation: ModelPlanGeneration::first(),
    entry_count: 0,
    slots: FixedArray::make(limits.max_plan_rows(), slot),
    request_ids: FixedArray::make(limits.max_plan_rows(), request_id),
    request_generations: FixedArray::make(limits.max_plan_rows(), generation),
    kinds: FixedArray::make(limits.max_plan_rows(), FailedCompletion),
    values: FixedArray::make(limits.max_plan_rows(), 0),
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
  if processed_tokens <= 0 || processed_tokens > self.limits.max_plan_tokens() {
    raise CompletionTokenCountMismatch(index=self.entry_count)
  }
  self.append(
    slot,
    request_id,
    request_generation,
    PrefillCompletion,
    processed_tokens,
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
  if token_id < 0 || token_id > self.limits.max_token_id() {
    raise InvalidCompletionToken(slot=slot.value())
  }
  self.append(
    slot,
    request_id,
    request_generation,
    DecodeCompletion,
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
  self.append(
    slot,
    request_id,
    request_generation,
    FailedCompletion,
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
  self.require_state(CompletionWritable)
  if self.entry_count == 0 {
    raise CompletionCountMismatch
  }
  self.validate_against(plan)
  self.sequence = plan.sequence()
  self.model_plan_generation = plan.model_plan_generation()
  self.state = CompletionSubmitted
  SubmittedCompletion(self, self.epoch)
}

///|
pub fn CompletionBuffer::consume(
  self : CompletionBuffer,
  submitted : SubmittedCompletion,
  plan : SubmittedSchedulePlan,
) -> ConsumedCompletion raise WorkerProtocolError {
  submitted.authenticate_owner(self)
  submitted.validate_for(plan)
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
  self.entry_count = 0
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
