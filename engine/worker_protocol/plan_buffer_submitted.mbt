///|
fn SubmittedSchedulePlan::authenticate(
  self : SubmittedSchedulePlan,
) -> SchedulePlanBuffer raise WorkerProtocolError {
  let owner = self.0
  if self.1 != owner.epoch {
    raise StalePlanBufferEpoch
  }
  owner.require_state(Submitted)
  owner
}

///|
/// Returns the opaque handle for this owner's exact current submitted epoch.
/// The handle is stack-valued and does not expose mutable owner access.
pub fn SchedulePlanBuffer::current_submitted(
  self : SchedulePlanBuffer,
) -> SubmittedSchedulePlan raise WorkerProtocolError {
  self.require_state(Submitted)
  SubmittedSchedulePlan(self, self.epoch)
}

///|
/// Returns the stable payload-free identity of this submission's exact owner
/// after reauthenticating its current epoch and lifecycle.
pub fn SubmittedSchedulePlan::buffer_identity(
  self : SubmittedSchedulePlan,
) -> PlanBufferIdentity raise WorkerProtocolError {
  self.authenticate().identity
}

///|
fn SubmittedSchedulePlan::authenticate_owner(
  self : SubmittedSchedulePlan,
  owner : SchedulePlanBuffer,
) -> Unit raise WorkerProtocolError {
  if !physical_equal(self.0, owner) {
    raise PlanBufferMismatch
  }
  ignore(self.authenticate())
}

///|
fn check_table_index(
  table : ProtocolTable,
  index : Int,
  length : Int,
) -> Unit raise WorkerProtocolError {
  if index < 0 || index >= length {
    raise InvalidView(table~, rule=RangeOutsideTable)
  }
}

///|
fn check_row_index(
  kind : WorkerRowKind,
  index : Int,
  length : Int,
) -> Unit raise WorkerProtocolError {
  if index < 0 || index >= length {
    row_error(kind, index, RangeOutsideTable)
  }
}

///|
pub fn SubmittedSchedulePlan::sequence(
  self : SubmittedSchedulePlan,
) -> PlanSequence raise WorkerProtocolError {
  let owner = self.authenticate()
  try! PlanSequence::from_wire(owner.sequence_value)
}

///|
/// Returns the authenticated scalar plan sequence without constructing an
/// identity wrapper.
pub fn SubmittedSchedulePlan::sequence_wire(
  self : SubmittedSchedulePlan,
) -> UInt64 raise WorkerProtocolError {
  self.authenticate().sequence_value
}

///|
pub fn SubmittedSchedulePlan::model_plan_generation(
  self : SubmittedSchedulePlan,
) -> ModelPlanGeneration raise WorkerProtocolError {
  let owner = self.authenticate()
  try! ModelPlanGeneration::from_wire(owner.model_plan_generation_value)
}

///|
/// Returns the authenticated scalar model-plan generation without constructing
/// an identity wrapper.
pub fn SubmittedSchedulePlan::model_plan_generation_wire(
  self : SubmittedSchedulePlan,
) -> UInt64 raise WorkerProtocolError {
  self.authenticate().model_plan_generation_value
}

///|
pub fn SubmittedSchedulePlan::token_budget(
  self : SubmittedSchedulePlan,
) -> Int raise WorkerProtocolError {
  self.authenticate().token_budget
}

///|
pub fn SubmittedSchedulePlan::token_count(
  self : SubmittedSchedulePlan,
) -> Int raise WorkerProtocolError {
  self.authenticate().token_count
}

///|
pub fn SubmittedSchedulePlan::page_count(
  self : SubmittedSchedulePlan,
) -> Int raise WorkerProtocolError {
  self.authenticate().page_count
}

///|
pub fn SubmittedSchedulePlan::capability_count(
  self : SubmittedSchedulePlan,
) -> Int raise WorkerProtocolError {
  self.authenticate().capability_count
}

///|
pub fn SubmittedSchedulePlan::prefill_count(
  self : SubmittedSchedulePlan,
) -> Int raise WorkerProtocolError {
  self.authenticate().prefill_count
}

///|
pub fn SubmittedSchedulePlan::decode_count(
  self : SubmittedSchedulePlan,
) -> Int raise WorkerProtocolError {
  self.authenticate().decode_count
}

///|
pub fn SubmittedSchedulePlan::row_count(
  self : SubmittedSchedulePlan,
) -> Int raise WorkerProtocolError {
  let owner = self.authenticate()
  owner.prefill_count + owner.decode_count
}

///|
pub fn SubmittedSchedulePlan::token_at(
  self : SubmittedSchedulePlan,
  index : Int,
) -> Int raise WorkerProtocolError {
  let owner = self.authenticate()
  check_table_index(Tokens, index, owner.token_count)
  owner.tokens[index]
}

///|
pub fn SubmittedSchedulePlan::page_at(
  self : SubmittedSchedulePlan,
  index : Int,
) -> @page_allocator.PageId raise WorkerProtocolError {
  let owner = self.authenticate()
  check_table_index(Pages, index, owner.page_count)
  try owner.pages.read(index) catch {
    _ => raise PlanBufferStorageFailure
  } noraise {
    page => page
  }
}

///|
pub fn SubmittedSchedulePlan::capability_at(
  self : SubmittedSchedulePlan,
  index : Int,
) -> @plan.KernelCapabilityId raise WorkerProtocolError {
  let owner = self.authenticate()
  check_table_index(Capabilities, index, owner.capability_count)
  owner.capabilities[index]
}

///|
pub fn SubmittedSchedulePlan::prefill_row(
  self : SubmittedSchedulePlan,
  index : Int,
) -> SubmittedPrefillRow raise WorkerProtocolError {
  let owner = self.authenticate()
  check_row_index(Prefill, index, owner.prefill_count)
  SubmittedPrefillRow(self.0, self.1, index)
}

///|
pub fn SubmittedSchedulePlan::decode_row(
  self : SubmittedSchedulePlan,
  index : Int,
) -> SubmittedDecodeRow raise WorkerProtocolError {
  let owner = self.authenticate()
  check_row_index(Decode, index, owner.decode_count)
  SubmittedDecodeRow(self.0, self.1, index)
}

///|
fn SubmittedPrefillRow::owner(
  self : SubmittedPrefillRow,
) -> SchedulePlanBuffer raise WorkerProtocolError {
  let owner = SubmittedSchedulePlan(self.0, self.1).authenticate()
  check_row_index(Prefill, self.2, owner.prefill_count)
  owner
}

///|
pub fn SubmittedPrefillRow::request_id(
  self : SubmittedPrefillRow,
) -> @inference.RequestId raise WorkerProtocolError {
  self.owner().prefill_request_ids[self.2]
}

///|
pub fn SubmittedPrefillRow::request_generation(
  self : SubmittedPrefillRow,
) -> @inference.RequestGeneration raise WorkerProtocolError {
  self.owner().prefill_request_generations[self.2]
}

///|
pub fn SubmittedPrefillRow::sequence_token_start(
  self : SubmittedPrefillRow,
) -> Int raise WorkerProtocolError {
  self.owner().prefill_sequence_starts[self.2]
}

///|
pub fn SubmittedPrefillRow::tokens(
  self : SubmittedPrefillRow,
) -> TokenRange raise WorkerProtocolError {
  let owner = self.owner()
  TokenRange(
    owner.prefill_token_starts[self.2],
    owner.prefill_token_lengths[self.2],
  )
}

///|
pub fn SubmittedPrefillRow::page_table(
  self : SubmittedPrefillRow,
) -> PageTableView raise WorkerProtocolError {
  let owner = self.owner()
  PageTableView(
    owner.prefill_page_starts[self.2],
    owner.prefill_page_lengths[self.2],
  )
}

///|
pub fn SubmittedPrefillRow::capabilities(
  self : SubmittedPrefillRow,
) -> CapabilityView raise WorkerProtocolError {
  let owner = self.owner()
  CapabilityView(
    owner.prefill_capability_starts[self.2],
    owner.prefill_capability_lengths[self.2],
  )
}

///|
pub fn SubmittedPrefillRow::produces_token(
  self : SubmittedPrefillRow,
) -> Bool raise WorkerProtocolError {
  self.owner().prefill_produces_tokens[self.2]
}

///|
pub fn SubmittedPrefillRow::sampling(
  self : SubmittedPrefillRow,
) -> @inference.SamplingParameters raise WorkerProtocolError {
  self.owner().prefill_sampling[self.2]
}

///|
pub fn SubmittedPrefillRow::sampling_seed_value(
  self : SubmittedPrefillRow,
) -> UInt64 raise WorkerProtocolError {
  self.owner().prefill_sampling_seed_values[self.2]
}

///|
pub fn SubmittedPrefillRow::sample_index(
  self : SubmittedPrefillRow,
) -> Int raise WorkerProtocolError {
  self.owner().prefill_sample_indices[self.2]
}

///|
pub fn SubmittedPrefillRow::completion_slot(
  self : SubmittedPrefillRow,
) -> CompletionSlot raise WorkerProtocolError {
  self.owner().prefill_completion_slots[self.2]
}

///|
fn SubmittedDecodeRow::owner(
  self : SubmittedDecodeRow,
) -> SchedulePlanBuffer raise WorkerProtocolError {
  let owner = SubmittedSchedulePlan(self.0, self.1).authenticate()
  check_row_index(Decode, self.2, owner.decode_count)
  owner
}

///|
pub fn SubmittedDecodeRow::request_id(
  self : SubmittedDecodeRow,
) -> @inference.RequestId raise WorkerProtocolError {
  self.owner().decode_request_ids[self.2]
}

///|
pub fn SubmittedDecodeRow::request_generation(
  self : SubmittedDecodeRow,
) -> @inference.RequestGeneration raise WorkerProtocolError {
  self.owner().decode_request_generations[self.2]
}

///|
pub fn SubmittedDecodeRow::sequence_token_position(
  self : SubmittedDecodeRow,
) -> Int raise WorkerProtocolError {
  self.owner().decode_sequence_positions[self.2]
}

///|
pub fn SubmittedDecodeRow::input_token(
  self : SubmittedDecodeRow,
) -> TokenRange raise WorkerProtocolError {
  let owner = self.owner()
  TokenRange(owner.decode_token_starts[self.2], 1)
}

///|
pub fn SubmittedDecodeRow::page_table(
  self : SubmittedDecodeRow,
) -> PageTableView raise WorkerProtocolError {
  let owner = self.owner()
  PageTableView(
    owner.decode_page_starts[self.2],
    owner.decode_page_lengths[self.2],
  )
}

///|
pub fn SubmittedDecodeRow::capabilities(
  self : SubmittedDecodeRow,
) -> CapabilityView raise WorkerProtocolError {
  let owner = self.owner()
  CapabilityView(
    owner.decode_capability_starts[self.2],
    owner.decode_capability_lengths[self.2],
  )
}

///|
pub fn SubmittedDecodeRow::sampling(
  self : SubmittedDecodeRow,
) -> @inference.SamplingParameters raise WorkerProtocolError {
  self.owner().decode_sampling[self.2]
}

///|
pub fn SubmittedDecodeRow::sampling_seed_value(
  self : SubmittedDecodeRow,
) -> UInt64 raise WorkerProtocolError {
  self.owner().decode_sampling_seed_values[self.2]
}

///|
pub fn SubmittedDecodeRow::sample_index(
  self : SubmittedDecodeRow,
) -> Int raise WorkerProtocolError {
  self.owner().decode_sample_indices[self.2]
}

///|
pub fn SubmittedDecodeRow::completion_slot(
  self : SubmittedDecodeRow,
) -> CompletionSlot raise WorkerProtocolError {
  self.owner().decode_completion_slots[self.2]
}

///|
pub fn SchedulePlanBuffer::state(self : SchedulePlanBuffer) -> PlanBufferState {
  self.state
}

///|
pub fn SchedulePlanBuffer::epoch(self : SchedulePlanBuffer) -> UInt64 {
  self.epoch
}
