///|
fn SchedulePlanBuffer::start_draft(
  self : SchedulePlanBuffer,
  kind : WorkerRowKind,
  request_id : @inference.RequestId,
  request_generation : @inference.RequestGeneration,
  sequence_position : Int,
  produces_token : Bool,
  sampling : @inference.SamplingParameters,
  completion_slot : CompletionSlot,
) -> PlanRowDraft raise WorkerProtocolError {
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
  match kind {
    Prefill =>
      if sequence_position < 0 ||
        sequence_position > self.limits.max_sequence_tokens() {
        raise InvalidRowDescriptor(kind=Prefill, rule=InvalidSequencePosition)
      }
    Decode =>
      if sequence_position < 0 ||
        sequence_position >= self.limits.max_sequence_tokens() {
        raise InvalidRowDescriptor(kind=Decode, rule=InvalidSequencePosition)
      }
  }
  if self.draft_epoch == 0xffffffffffffffffUL {
    raise IdentityOverflow(PlanRowDraftEpochIdentity)
  }
  self.draft_epoch = self.draft_epoch + 1UL
  self.draft_open = true
  self.draft_kind = kind
  self.draft_request_id = request_id
  self.draft_request_generation = request_generation
  self.draft_sequence_position = sequence_position
  self.draft_produces_token = produces_token
  self.draft_completion_slot = completion_slot
  self.draft_sampling = sampling
  self.draft_token_start = self.token_count
  self.draft_page_start = self.page_count
  self.draft_capability_start = self.capability_count
  PlanRowDraft(self, self.epoch, self.draft_epoch)
}

///|
/// Begins one scalar-built prefill row. The row is invisible until commit.
pub fn SchedulePlanBuffer::begin_prefill(
  self : SchedulePlanBuffer,
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  sequence_token_start~ : Int,
  completion_slot~ : CompletionSlot,
) -> PlanRowDraft raise WorkerProtocolError {
  self.start_draft(
    Prefill,
    request_id,
    request_generation,
    sequence_token_start,
    false,
    @inference.SamplingParameters::greedy(),
    completion_slot,
  )
}

///|
/// Begins the final prompt chunk, which samples exactly one output token.
pub fn SchedulePlanBuffer::begin_final_prefill(
  self : SchedulePlanBuffer,
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  sequence_token_start~ : Int,
  sampling~ : @inference.SamplingParameters,
  completion_slot~ : CompletionSlot,
) -> PlanRowDraft raise WorkerProtocolError {
  self.start_draft(
    Prefill,
    request_id,
    request_generation,
    sequence_token_start,
    true,
    sampling,
    completion_slot,
  )
}

///|
/// Begins one scalar-built decode row. Exactly one token is required at commit.
pub fn SchedulePlanBuffer::begin_decode(
  self : SchedulePlanBuffer,
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  sequence_token_position~ : Int,
  sampling~ : @inference.SamplingParameters,
  completion_slot~ : CompletionSlot,
) -> PlanRowDraft raise WorkerProtocolError {
  self.start_draft(
    Decode,
    request_id,
    request_generation,
    sequence_token_position,
    false,
    sampling,
    completion_slot,
  )
}

///|
fn PlanRowDraft::owner(
  self : PlanRowDraft,
) -> SchedulePlanBuffer raise WorkerProtocolError {
  let owner = self.0
  if self.1 != owner.epoch {
    raise StalePlanRowDraftEpoch
  }
  owner.require_state(Writable)
  if !owner.draft_open {
    raise PlanRowDraftNotOpen
  }
  if self.2 != owner.draft_epoch {
    raise PlanRowDraftMismatch
  }
  owner
}

///|
fn PlanRowDraft::authenticate_owner(
  self : PlanRowDraft,
  owner : SchedulePlanBuffer,
) -> Unit raise WorkerProtocolError {
  if !physical_equal(self.0, owner) {
    raise PlanRowDraftMismatch
  }
  ignore(self.owner())
}

///|
/// Owner-side commit rejects a draft originating from another plan buffer.
pub fn SchedulePlanBuffer::commit_draft(
  self : SchedulePlanBuffer,
  draft : PlanRowDraft,
) -> Unit raise WorkerProtocolError {
  draft.authenticate_owner(self)
  draft.commit()
}

///|
/// Owner-side rollback rejects a draft originating from another plan buffer.
pub fn SchedulePlanBuffer::rollback_draft(
  self : SchedulePlanBuffer,
  draft : PlanRowDraft,
) -> Unit raise WorkerProtocolError {
  draft.authenticate_owner(self)
  draft.rollback()
}

///|
pub fn PlanRowDraft::push_token(
  self : PlanRowDraft,
  token : Int,
) -> Unit raise WorkerProtocolError {
  let owner = self.owner()
  if owner.token_count >= owner.limits.max_plan_tokens() {
    raise BoundExceeded(PlanTokens)
  }
  if owner.draft_kind == Decode && owner.token_count > owner.draft_token_start {
    raise InvalidRowDescriptor(kind=Decode, rule=RangeOverflow)
  }
  if owner.draft_kind == Prefill &&
    owner.token_count - owner.draft_token_start >=
    owner.limits.max_sequence_tokens() - owner.draft_sequence_position {
    raise InvalidRowDescriptor(kind=Prefill, rule=InvalidSequencePosition)
  }
  validate_token(token, owner.token_count, owner.limits)
  owner.tokens[owner.token_count] = token
  owner.token_count = owner.token_count + 1
}

///|
pub fn PlanRowDraft::push_page(
  self : PlanRowDraft,
  page : @page_allocator.PageId,
) -> Unit raise WorkerProtocolError {
  let owner = self.owner()
  if owner.page_count >= owner.limits.max_plan_pages() {
    raise BoundExceeded(PlanPages)
  }
  map_page_storage_write(owner.pages, owner.page_count, page)
  owner.page_count = owner.page_count + 1
}

///|
pub fn PlanRowDraft::push_capability(
  self : PlanRowDraft,
  capability : @plan.KernelCapabilityId,
) -> Unit raise WorkerProtocolError {
  let owner = self.owner()
  if owner.capability_count >= owner.limits.max_capabilities() {
    raise BoundExceeded(Capabilities)
  }
  owner.capabilities[owner.capability_count] = capability
  owner.capability_count = owner.capability_count + 1
}

///|
fn PlanRowDraft::validate_complete(
  _self : PlanRowDraft,
  owner : SchedulePlanBuffer,
) -> Unit raise WorkerProtocolError {
  let token_length = owner.token_count - owner.draft_token_start
  let page_length = owner.page_count - owner.draft_page_start
  let capability_length = owner.capability_count - owner.draft_capability_start
  if token_length <= 0 || page_length <= 0 || capability_length <= 0 {
    raise InvalidRowDescriptor(kind=owner.draft_kind, rule=EmptyRange)
  }
  if owner.draft_kind == Decode && token_length != 1 {
    raise InvalidRowDescriptor(kind=Decode, rule=RangeOverflow)
  }
}

///|
/// Publishes the exact reserved table suffix as one row. A failed commit keeps
/// the draft open so the scheduler may push missing cells or roll it back.
pub fn PlanRowDraft::commit(
  self : PlanRowDraft,
) -> Unit raise WorkerProtocolError {
  let owner = self.owner()
  self.validate_complete(owner)
  let token_length = owner.token_count - owner.draft_token_start
  let page_length = owner.page_count - owner.draft_page_start
  let capability_length = owner.capability_count - owner.draft_capability_start
  match owner.draft_kind {
    Prefill => {
      let row = owner.prefill_count
      owner.prefill_request_ids[row] = owner.draft_request_id
      owner.prefill_request_generations[row] = owner.draft_request_generation
      owner.prefill_sequence_starts[row] = owner.draft_sequence_position
      owner.prefill_token_starts[row] = owner.draft_token_start
      owner.prefill_token_lengths[row] = token_length
      owner.prefill_page_starts[row] = owner.draft_page_start
      owner.prefill_page_lengths[row] = page_length
      owner.prefill_capability_starts[row] = owner.draft_capability_start
      owner.prefill_capability_lengths[row] = capability_length
      owner.prefill_produces_tokens[row] = owner.draft_produces_token
      owner.prefill_sampling[row] = owner.draft_sampling
      owner.prefill_completion_slots[row] = owner.draft_completion_slot
      owner.prefill_count = row + 1
    }
    Decode => {
      let row = owner.decode_count
      owner.decode_request_ids[row] = owner.draft_request_id
      owner.decode_request_generations[row] = owner.draft_request_generation
      owner.decode_sequence_positions[row] = owner.draft_sequence_position
      owner.decode_token_starts[row] = owner.draft_token_start
      owner.decode_page_starts[row] = owner.draft_page_start
      owner.decode_page_lengths[row] = page_length
      owner.decode_capability_starts[row] = owner.draft_capability_start
      owner.decode_capability_lengths[row] = capability_length
      owner.decode_sampling[row] = owner.draft_sampling
      owner.decode_completion_slots[row] = owner.draft_completion_slot
      owner.decode_count = row + 1
    }
  }
  owner.draft_open = false
}

///|
/// Discards every scalar table cell reserved since begin. No row was visible.
pub fn PlanRowDraft::rollback(
  self : PlanRowDraft,
) -> Unit raise WorkerProtocolError {
  let owner = self.owner()
  for index in owner.draft_page_start..<owner.page_count {
    map_page_storage_clear(owner.pages, index)
  }
  owner.token_count = owner.draft_token_start
  owner.page_count = owner.draft_page_start
  owner.capability_count = owner.draft_capability_start
  owner.draft_open = false
}
