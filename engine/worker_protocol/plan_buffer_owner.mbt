///|
fn map_page_storage_write(
  storage : @page_allocator.PageIdStorage,
  index : Int,
  page : @page_allocator.PageId,
) -> Unit raise WorkerProtocolError {
  try storage.write(index, page) catch {
    _ => raise PlanBufferStorageFailure
  } noraise {
    _ => ()
  }
}

///|
fn map_page_storage_clear(
  storage : @page_allocator.PageIdStorage,
  index : Int,
) -> Unit raise WorkerProtocolError {
  try storage.clear(index) catch {
    _ => raise PlanBufferStorageFailure
  } noraise {
    _ => ()
  }
}

///|
fn SchedulePlanBuffer::require_state(
  self : SchedulePlanBuffer,
  expected : PlanBufferState,
) -> Unit raise WorkerProtocolError {
  if self.state != expected {
    raise InvalidPlanBufferState(expected~, actual=self.state)
  }
}

///|
fn SchedulePlanBuffer::has_request(
  self : SchedulePlanBuffer,
  request_id : @inference.RequestId,
) -> Bool {
  for index in 0..<self.prefill_count {
    if self.prefill_request_ids[index] == request_id {
      return true
    }
  }
  for index in 0..<self.decode_count {
    if self.decode_request_ids[index] == request_id {
      return true
    }
  }
  false
}

///|
fn SchedulePlanBuffer::has_slot(
  self : SchedulePlanBuffer,
  completion_slot : CompletionSlot,
) -> Bool {
  for index in 0..<self.prefill_count {
    if self.prefill_completion_slots[index] == completion_slot {
      return true
    }
  }
  for index in 0..<self.decode_count {
    if self.decode_completion_slots[index] == completion_slot {
      return true
    }
  }
  false
}

///|
fn SchedulePlanBuffer::preflight_row(
  self : SchedulePlanBuffer,
  kind : WorkerRowKind,
  request_id : @inference.RequestId,
  completion_slot : CompletionSlot,
  token_length : Int,
  page_length : Int,
  capability_length : Int,
) -> Unit raise WorkerProtocolError {
  self.require_state(Writable)
  if self.draft_open {
    raise PlanRowDraftAlreadyOpen
  }
  if self.prefill_count + self.decode_count >= self.limits.max_plan_rows() {
    raise BoundExceeded(PlanRows)
  }
  match kind {
    Prefill if self.decode_count > 0 =>
      raise InvalidRowDescriptor(kind=Prefill, rule=InvalidRowOrder)
    Prefill if self.prefill_count >= self.limits.max_prefill_rows() =>
      raise BoundExceeded(PrefillRows)
    Decode if self.decode_count >= self.limits.max_decode_rows() =>
      raise BoundExceeded(DecodeRows)
    _ => ()
  }
  if token_length <= 0 || page_length <= 0 || capability_length <= 0 {
    raise InvalidRowDescriptor(kind~, rule=EmptyRange)
  }
  if token_length > self.limits.max_plan_tokens() - self.token_count {
    raise BoundExceeded(PlanTokens)
  }
  if page_length > self.limits.max_plan_pages() - self.page_count {
    raise BoundExceeded(PlanPages)
  }
  if capability_length > self.limits.max_capabilities() - self.capability_count {
    raise BoundExceeded(Capabilities)
  }
  if completion_slot.value() < 0 ||
    completion_slot.value() >= self.limits.max_completion_slots() {
    raise InvalidCompletionSlot
  }
  if self.has_request(request_id) {
    raise InvalidRowDescriptor(kind~, rule=DuplicateRequest)
  }
  if self.has_slot(completion_slot) {
    raise InvalidRowDescriptor(kind~, rule=DuplicateCompletionSlot)
  }
}

///|
fn SchedulePlanBuffer::copy_tables(
  self : SchedulePlanBuffer,
  tokens : ArrayView[Int],
  pages : ArrayView[@page_allocator.PageId],
  capabilities : ArrayView[@plan.KernelCapabilityId],
) -> Unit raise WorkerProtocolError {
  for index, token in tokens {
    self.tokens[self.token_count + index] = token
  }
  for index, page in pages {
    map_page_storage_write(self.pages, self.page_count + index, page)
  }
  for index, capability in capabilities {
    self.capabilities[self.capability_count + index] = capability
  }
}

///|
/// Allocates every table and row column once. Successful append, submit,
/// access, retire, and reset operations never resize a collection.
pub fn SchedulePlanBuffer::new(
  limits : WorkerProtocolLimits,
) -> SchedulePlanBuffer raise WorkerProtocolError {
  let request_id = try! @inference.RequestId::new(1UL)
  let request_generation = @inference.RequestGeneration::initial()
  let completion_slot = CompletionSlot::new(0, limits)
  let sampling = @inference.SamplingParameters::greedy()
  let pages = try
    @page_allocator.PageIdStorage::new(limits.max_plan_pages())
  catch {
    _ => raise PlanBufferStorageFailure
  } noraise {
    storage => storage
  }
  {
    limits,
    state: Writable,
    epoch: 1UL,
    sequence_value: 1UL,
    model_plan_generation_value: 1UL,
    token_budget: 0,
    token_count: 0,
    page_count: 0,
    capability_count: 0,
    prefill_count: 0,
    decode_count: 0,
    draft_open: false,
    draft_epoch: 0UL,
    checkpoint_epoch: 0UL,
    checkpoint_token_count: 0,
    checkpoint_page_count: 0,
    checkpoint_capability_count: 0,
    checkpoint_prefill_count: 0,
    checkpoint_decode_count: 0,
    draft_kind: Prefill,
    draft_request_id: request_id,
    draft_request_generation: request_generation,
    draft_sequence_position: 0,
    draft_produces_token: false,
    draft_completion_slot: completion_slot,
    draft_sampling: sampling,
    draft_token_start: 0,
    draft_page_start: 0,
    draft_capability_start: 0,
    tokens: FixedArray::make(limits.max_plan_tokens(), 0),
    pages,
    capabilities: FixedArray::make(
      limits.max_capabilities(),
      @plan.KernelCapabilityId::embedding_lookup(),
    ),
    prefill_request_ids: FixedArray::make(limits.max_prefill_rows(), request_id),
    prefill_request_generations: FixedArray::make(
      limits.max_prefill_rows(),
      request_generation,
    ),
    prefill_sequence_starts: FixedArray::make(limits.max_prefill_rows(), 0),
    prefill_token_starts: FixedArray::make(limits.max_prefill_rows(), 0),
    prefill_token_lengths: FixedArray::make(limits.max_prefill_rows(), 0),
    prefill_page_starts: FixedArray::make(limits.max_prefill_rows(), 0),
    prefill_page_lengths: FixedArray::make(limits.max_prefill_rows(), 0),
    prefill_capability_starts: FixedArray::make(limits.max_prefill_rows(), 0),
    prefill_capability_lengths: FixedArray::make(limits.max_prefill_rows(), 0),
    prefill_produces_tokens: FixedArray::make(limits.max_prefill_rows(), false),
    prefill_sampling: FixedArray::make(limits.max_prefill_rows(), sampling),
    prefill_completion_slots: FixedArray::make(
      limits.max_prefill_rows(),
      completion_slot,
    ),
    decode_request_ids: FixedArray::make(limits.max_decode_rows(), request_id),
    decode_request_generations: FixedArray::make(
      limits.max_decode_rows(),
      request_generation,
    ),
    decode_sequence_positions: FixedArray::make(limits.max_decode_rows(), 0),
    decode_token_starts: FixedArray::make(limits.max_decode_rows(), 0),
    decode_page_starts: FixedArray::make(limits.max_decode_rows(), 0),
    decode_page_lengths: FixedArray::make(limits.max_decode_rows(), 0),
    decode_capability_starts: FixedArray::make(limits.max_decode_rows(), 0),
    decode_capability_lengths: FixedArray::make(limits.max_decode_rows(), 0),
    decode_sampling: FixedArray::make(limits.max_decode_rows(), sampling),
    decode_completion_slots: FixedArray::make(
      limits.max_decode_rows(),
      completion_slot,
    ),
  }
}

///|
fn SchedulePlanBuffer::validate_contents(
  self : SchedulePlanBuffer,
  token_budget : Int,
) -> Unit raise WorkerProtocolError {
  validate_plan_counts(
    self.prefill_count,
    self.decode_count,
    self.token_count,
    self.page_count,
    self.capability_count,
    self.limits,
  )
  for index in 0..<self.token_count {
    validate_token(self.tokens[index], index, self.limits)
  }
  let mut token_end = 0
  let mut page_end = 0
  let mut capability_end = 0
  let mut scheduled_tokens = 0
  for index in 0..<self.prefill_count {
    token_end = validate_actual_range(
      Prefill,
      index,
      self.prefill_token_starts[index],
      self.prefill_token_lengths[index],
      self.token_count,
      token_end,
    )
    page_end = validate_actual_range(
      Prefill,
      index,
      self.prefill_page_starts[index],
      self.prefill_page_lengths[index],
      self.page_count,
      page_end,
    )
    capability_end = validate_actual_range(
      Prefill,
      index,
      self.prefill_capability_starts[index],
      self.prefill_capability_lengths[index],
      self.capability_count,
      capability_end,
    )
    scheduled_tokens = scheduled_tokens + self.prefill_token_lengths[index]
  }
  for index in 0..<self.decode_count {
    token_end = validate_actual_range(
      Decode,
      index,
      self.decode_token_starts[index],
      1,
      self.token_count,
      token_end,
    )
    page_end = validate_actual_range(
      Decode,
      index,
      self.decode_page_starts[index],
      self.decode_page_lengths[index],
      self.page_count,
      page_end,
    )
    capability_end = validate_actual_range(
      Decode,
      index,
      self.decode_capability_starts[index],
      self.decode_capability_lengths[index],
      self.capability_count,
      capability_end,
    )
    scheduled_tokens = scheduled_tokens + 1
  }
  validate_plan_finish(
    token_end,
    self.token_count,
    page_end,
    self.page_count,
    capability_end,
    self.capability_count,
    scheduled_tokens,
    token_budget,
    self.limits,
  )
}

///|
/// Authenticates exact worker order and loaded model generation, then freezes
/// the current epoch without copying any table or row.
pub fn SchedulePlanBuffer::submit(
  self : SchedulePlanBuffer,
  sequence~ : PlanSequence,
  model_plan_generation~ : ModelPlanGeneration,
  token_budget~ : Int,
  previous_sequence : PlanSequence?,
  loaded_generation : ModelPlanGeneration,
) -> SubmittedSchedulePlan raise WorkerProtocolError {
  self.submit_wire(
    sequence_value=sequence.value(),
    model_plan_generation_value=model_plan_generation.value(),
    token_budget~,
    has_previous=previous_sequence is Some(_),
    previous_sequence_value=previous_sequence
      .map(PlanSequence::value)
      .unwrap_or(0UL),
    loaded_generation_value=loaded_generation.value(),
  )
}

///|
/// Scalar submission seam for allocation-sensitive schedulers. It validates
/// the same identities as `submit`, stores wire scalars directly, and freezes
/// the current epoch without constructing transient identity wrappers.
pub fn SchedulePlanBuffer::submit_wire(
  self : SchedulePlanBuffer,
  sequence_value~ : UInt64,
  model_plan_generation_value~ : UInt64,
  token_budget~ : Int,
  has_previous~ : Bool,
  previous_sequence_value~ : UInt64,
  loaded_generation_value~ : UInt64,
) -> SubmittedSchedulePlan raise WorkerProtocolError {
  self.require_state(Writable)
  if self.draft_open {
    raise PlanRowDraftAlreadyOpen
  }
  self.validate_contents(token_budget)
  validate_submission_identity_wire(
    sequence_value, model_plan_generation_value, has_previous, previous_sequence_value,
    loaded_generation_value,
  )
  self.sequence_value = sequence_value
  self.model_plan_generation_value = model_plan_generation_value
  self.token_budget = token_budget
  self.state = Submitted
  SubmittedSchedulePlan(self, self.epoch)
}

///|
/// Authenticates and retires exactly the submitted epoch. Retiring prevents
/// further worker reads before any storage can be reused.
///
/// Calling this method asserts that the scheduler has already authenticated a
/// matching worker completion or otherwise proved device work retired. The
/// in-memory protocol cannot prove external worker progress by itself.
pub fn SchedulePlanBuffer::retire(
  self : SchedulePlanBuffer,
  submitted : SubmittedSchedulePlan,
) -> RetiredSchedulePlan raise WorkerProtocolError {
  submitted.authenticate_owner(self)
  self.state = Retired
  RetiredSchedulePlan(self, self.epoch)
}

///|
fn RetiredSchedulePlan::authenticate_owner(
  self : RetiredSchedulePlan,
  owner : SchedulePlanBuffer,
) -> Unit raise WorkerProtocolError {
  if !physical_equal(self.0, owner) {
    raise PlanBufferMismatch
  }
  if self.1 != owner.epoch {
    raise StalePlanBufferEpoch
  }
  owner.require_state(Retired)
}

///|
/// Clears the logical plan and advances the epoch. Reset is legal only with
/// the exact retirement proof; old submitted and row handles then stay stale.
pub fn SchedulePlanBuffer::reset(
  self : SchedulePlanBuffer,
  retired : RetiredSchedulePlan,
) -> Unit raise WorkerProtocolError {
  retired.authenticate_owner(self)
  if self.epoch == 0xffffffffffffffffUL {
    raise IdentityOverflow(PlanBufferEpochIdentity)
  }
  for index in 0..<self.page_count {
    map_page_storage_clear(self.pages, index)
  }
  self.token_budget = 0
  self.token_count = 0
  self.page_count = 0
  self.capability_count = 0
  self.prefill_count = 0
  self.decode_count = 0
  self.draft_open = false
  self.checkpoint_token_count = 0
  self.checkpoint_page_count = 0
  self.checkpoint_capability_count = 0
  self.checkpoint_prefill_count = 0
  self.checkpoint_decode_count = 0
  self.epoch = self.epoch + 1UL
  self.state = Writable
}
