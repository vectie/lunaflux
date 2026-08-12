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
    sequence: PlanSequence::first(),
    model_plan_generation: ModelPlanGeneration::first(),
    token_budget: 0,
    token_count: 0,
    page_count: 0,
    capability_count: 0,
    prefill_count: 0,
    decode_count: 0,
    draft_open: false,
    draft_epoch: 0UL,
    draft_kind: Prefill,
    draft_request_id: request_id,
    draft_request_generation: request_generation,
    draft_sequence_position: 0,
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
/// Appends one exact prefill row and copies its flattened table segments into
/// preallocated storage. All validation precedes mutation.
pub fn SchedulePlanBuffer::append_prefill(
  self : SchedulePlanBuffer,
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  sequence_token_start~ : Int,
  tokens~ : ArrayView[Int],
  pages~ : ArrayView[@page_allocator.PageId],
  capabilities~ : ArrayView[@plan.KernelCapabilityId],
  completion_slot~ : CompletionSlot,
) -> Unit raise WorkerProtocolError {
  self.preflight_row(
    Prefill,
    request_id,
    completion_slot,
    tokens.length(),
    pages.length(),
    capabilities.length(),
  )
  if sequence_token_start < 0 ||
    sequence_token_start > self.limits.max_sequence_tokens() ||
    tokens.length() > self.limits.max_sequence_tokens() - sequence_token_start {
    raise InvalidRowDescriptor(kind=Prefill, rule=InvalidSequencePosition)
  }
  for index, token in tokens {
    validate_token(token, self.token_count + index, self.limits)
  }
  let row = self.prefill_count
  let token_start = self.token_count
  let page_start = self.page_count
  let capability_start = self.capability_count
  self.copy_tables(tokens, pages, capabilities)
  self.prefill_request_ids[row] = request_id
  self.prefill_request_generations[row] = request_generation
  self.prefill_sequence_starts[row] = sequence_token_start
  self.prefill_token_starts[row] = token_start
  self.prefill_token_lengths[row] = tokens.length()
  self.prefill_page_starts[row] = page_start
  self.prefill_page_lengths[row] = pages.length()
  self.prefill_capability_starts[row] = capability_start
  self.prefill_capability_lengths[row] = capabilities.length()
  self.prefill_completion_slots[row] = completion_slot
  self.token_count = token_start + tokens.length()
  self.page_count = page_start + pages.length()
  self.capability_count = capability_start + capabilities.length()
  self.prefill_count = row + 1
}

///|
/// Appends one exact decode row. The single input token and all table segments
/// are copied only after every bound and duplicate check succeeds.
pub fn SchedulePlanBuffer::append_decode(
  self : SchedulePlanBuffer,
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  sequence_token_position~ : Int,
  input_token~ : Int,
  pages~ : ArrayView[@page_allocator.PageId],
  capabilities~ : ArrayView[@plan.KernelCapabilityId],
  sampling~ : @inference.SamplingParameters,
  completion_slot~ : CompletionSlot,
) -> Unit raise WorkerProtocolError {
  self.preflight_row(
    Decode,
    request_id,
    completion_slot,
    1,
    pages.length(),
    capabilities.length(),
  )
  if sequence_token_position < 0 ||
    sequence_token_position >= self.limits.max_sequence_tokens() {
    raise InvalidRowDescriptor(kind=Decode, rule=InvalidSequencePosition)
  }
  validate_token(input_token, self.token_count, self.limits)
  let row = self.decode_count
  let token_start = self.token_count
  let page_start = self.page_count
  let capability_start = self.capability_count
  self.tokens[token_start] = input_token
  for index, page in pages {
    map_page_storage_write(self.pages, page_start + index, page)
  }
  for index, capability in capabilities {
    self.capabilities[capability_start + index] = capability
  }
  self.decode_request_ids[row] = request_id
  self.decode_request_generations[row] = request_generation
  self.decode_sequence_positions[row] = sequence_token_position
  self.decode_token_starts[row] = token_start
  self.decode_page_starts[row] = page_start
  self.decode_page_lengths[row] = pages.length()
  self.decode_capability_starts[row] = capability_start
  self.decode_capability_lengths[row] = capabilities.length()
  self.decode_sampling[row] = sampling
  self.decode_completion_slots[row] = completion_slot
  self.token_count = token_start + 1
  self.page_count = page_start + pages.length()
  self.capability_count = capability_start + capabilities.length()
  self.decode_count = row + 1
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
  self.require_state(Writable)
  if self.draft_open {
    raise PlanRowDraftAlreadyOpen
  }
  self.validate_contents(token_budget)
  validate_submission_identity(
    sequence, model_plan_generation, previous_sequence, loaded_generation,
  )
  self.sequence = sequence
  self.model_plan_generation = model_plan_generation
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
  self.epoch = self.epoch + 1UL
  self.state = Writable
}
