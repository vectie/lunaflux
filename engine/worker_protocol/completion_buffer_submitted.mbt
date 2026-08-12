///|
fn CompletionBuffer::find_slot(
  self : CompletionBuffer,
  slot : CompletionSlot,
) -> Int {
  for index in 0..<self.entry_count {
    if self.slots[index] == slot {
      return index
    }
  }
  -1
}

///|
fn CompletionBuffer::validate_identity(
  self : CompletionBuffer,
  index : Int,
  request_id : @inference.RequestId,
  request_generation : @inference.RequestGeneration,
  row_index : Int,
) -> Unit raise WorkerProtocolError {
  if self.request_ids[index] != request_id ||
    self.request_generations[index] != request_generation {
    raise CompletionIdentityMismatch(index=row_index)
  }
}

///|
fn CompletionBuffer::validate_against(
  self : CompletionBuffer,
  plan : SubmittedSchedulePlan,
) -> Unit raise WorkerProtocolError {
  if self.entry_count != plan.row_count() {
    raise CompletionCountMismatch
  }
  let prefill_count = plan.prefill_count()
  for row_index in 0..<prefill_count {
    let row = plan.prefill_row(row_index)
    let index = self.find_slot(row.completion_slot())
    if index < 0 {
      raise UnknownCompletionSlot(index=row_index)
    }
    self.validate_identity(
      index,
      row.request_id(),
      row.request_generation(),
      row_index,
    )
    match self.kinds[index] {
      PrefillCompletion => {
        if row.produces_token() {
          raise CompletionKindMismatch(index=row_index)
        }
        if self.processed_token_counts[index] != row.tokens().length() {
          raise CompletionTokenCountMismatch(index=row_index)
        }
      }
      FinalPrefillCompletion => {
        if !row.produces_token() {
          raise CompletionKindMismatch(index=row_index)
        }
        if self.processed_token_counts[index] != row.tokens().length() {
          raise CompletionTokenCountMismatch(index=row_index)
        }
      }
      FailedCompletion => ()
      DecodeCompletion => raise CompletionKindMismatch(index=row_index)
    }
  }
  let decode_count = plan.decode_count()
  for decode_index in 0..<decode_count {
    let row_index = prefill_count + decode_index
    let row = plan.decode_row(decode_index)
    let index = self.find_slot(row.completion_slot())
    if index < 0 {
      raise UnknownCompletionSlot(index=row_index)
    }
    self.validate_identity(
      index,
      row.request_id(),
      row.request_generation(),
      row_index,
    )
    match self.kinds[index] {
      DecodeCompletion | FailedCompletion => ()
      PrefillCompletion | FinalPrefillCompletion =>
        raise CompletionKindMismatch(index=row_index)
    }
  }
}

///|
fn SubmittedCompletion::authenticate(
  self : SubmittedCompletion,
) -> CompletionBuffer raise WorkerProtocolError {
  let owner = self.0
  if self.1 != owner.epoch {
    raise StaleCompletionBufferEpoch
  }
  owner.require_state(CompletionSubmitted)
  owner
}

///|
fn SubmittedCompletion::authenticate_owner(
  self : SubmittedCompletion,
  owner : CompletionBuffer,
) -> Unit raise WorkerProtocolError {
  if !physical_equal(self.0, owner) {
    raise CompletionBufferMismatch
  }
  ignore(self.authenticate())
}

///|
pub fn SubmittedCompletion::sequence(
  self : SubmittedCompletion,
) -> PlanSequence raise WorkerProtocolError {
  self.authenticate().sequence
}

///|
pub fn SubmittedCompletion::model_plan_generation(
  self : SubmittedCompletion,
) -> ModelPlanGeneration raise WorkerProtocolError {
  self.authenticate().model_plan_generation
}

///|
pub fn SubmittedCompletion::entry_count(
  self : SubmittedCompletion,
) -> Int raise WorkerProtocolError {
  self.authenticate().entry_count
}

///|
pub fn SubmittedCompletion::entry(
  self : SubmittedCompletion,
  index : Int,
) -> SubmittedCompletionEntry raise WorkerProtocolError {
  let owner = self.authenticate()
  if index < 0 || index >= owner.entry_count {
    raise UnknownCompletionSlot(index~)
  }
  SubmittedCompletionEntry(owner, self.1, index)
}

///|
/// Reauthenticates the exact plan identity and every completion slot, request
/// generation, row kind, and processed-token count without copying entries.
pub fn SubmittedCompletion::validate_for(
  self : SubmittedCompletion,
  plan : SubmittedSchedulePlan,
) -> Unit raise WorkerProtocolError {
  let owner = self.authenticate()
  if owner.sequence != plan.sequence() {
    raise StalePlanSequence
  }
  if owner.model_plan_generation != plan.model_plan_generation() {
    raise StaleModelPlanGeneration
  }
  owner.validate_against(plan)
}

///|
fn SubmittedCompletionEntry::owner(
  self : SubmittedCompletionEntry,
) -> CompletionBuffer raise WorkerProtocolError {
  let owner = SubmittedCompletion(self.0, self.1).authenticate()
  if self.2 < 0 || self.2 >= owner.entry_count {
    raise UnknownCompletionSlot(index=self.2)
  }
  owner
}

///|
pub fn SubmittedCompletionEntry::slot(
  self : SubmittedCompletionEntry,
) -> CompletionSlot raise WorkerProtocolError {
  self.owner().slots[self.2]
}

///|
pub fn SubmittedCompletionEntry::request_id(
  self : SubmittedCompletionEntry,
) -> @inference.RequestId raise WorkerProtocolError {
  self.owner().request_ids[self.2]
}

///|
pub fn SubmittedCompletionEntry::request_generation(
  self : SubmittedCompletionEntry,
) -> @inference.RequestGeneration raise WorkerProtocolError {
  self.owner().request_generations[self.2]
}

///|
pub fn SubmittedCompletionEntry::kind(
  self : SubmittedCompletionEntry,
) -> CompletionEntryKind raise WorkerProtocolError {
  self.owner().kinds[self.2]
}

///|
pub fn SubmittedCompletionEntry::processed_tokens(
  self : SubmittedCompletionEntry,
) -> Int raise WorkerProtocolError {
  let owner = self.owner()
  match owner.kinds[self.2] {
    PrefillCompletion | FinalPrefillCompletion => ()
    _ => raise CompletionKindMismatch(index=self.2)
  }
  owner.processed_token_counts[self.2]
}

///|
pub fn SubmittedCompletionEntry::token_id(
  self : SubmittedCompletionEntry,
) -> Int raise WorkerProtocolError {
  let owner = self.owner()
  match owner.kinds[self.2] {
    FinalPrefillCompletion | DecodeCompletion => ()
    _ => raise CompletionKindMismatch(index=self.2)
  }
  owner.token_ids[self.2]
}

///|
pub fn SubmittedCompletionEntry::failure(
  self : SubmittedCompletionEntry,
) -> WorkerFailure raise WorkerProtocolError {
  let owner = self.owner()
  if owner.kinds[self.2] != FailedCompletion {
    raise CompletionKindMismatch(index=self.2)
  }
  owner.failures[self.2]
}

///|
/// Publication remains separate from plan authentication. A cancellation may
/// advance the scheduler-owned generation after device work was submitted.
pub fn SubmittedCompletionEntry::is_current(
  self : SubmittedCompletionEntry,
  request_id : @inference.RequestId,
  current_generation : @inference.RequestGeneration,
) -> Bool raise WorkerProtocolError {
  let owner = self.owner()
  owner.request_ids[self.2] == request_id &&
  owner.request_generations[self.2] == current_generation
}
