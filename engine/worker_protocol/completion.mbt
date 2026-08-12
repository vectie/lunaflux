///|
/// Backend-neutral bounded failure category for one row. Raw device or vendor
/// diagnostics stay behind the worker boundary.
pub(all) enum WorkerFailure {
  ExecutionFailed
  ResourceUnavailable
  CancelledBeforeExecution
} derive(Debug, Eq)

///|
/// Typed result for exactly one completion slot. It deliberately has no
/// `Debug` implementation because a success may contain a generated token ID.
pub(all) enum CompletionOutcome {
  PrefillCompleted(processed_tokens~ : Int)
  DecodeCompleted(token_id~ : Int)
  Failed(WorkerFailure)
} derive(Eq)

///|
/// One worker result tied to a request cancellation generation and slot. It
/// deliberately has no `Debug` implementation.
pub struct CompletionEntry {
  slot : CompletionSlot
  request_id : @inference.RequestId
  request_generation : @inference.RequestGeneration
  outcome : CompletionOutcome
} derive(Eq)

///|
pub fn CompletionEntry::prefill_completed(
  slot~ : CompletionSlot,
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  processed_tokens~ : Int,
  limits~ : WorkerProtocolLimits,
) -> CompletionEntry raise WorkerProtocolError {
  if processed_tokens <= 0 || processed_tokens > limits.max_plan_tokens() {
    raise CompletionTokenCountMismatch(index=slot.value())
  }
  {
    slot,
    request_id,
    request_generation,
    outcome: PrefillCompleted(processed_tokens~),
  }
}

///|
pub fn CompletionEntry::decode_completed(
  slot~ : CompletionSlot,
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  token_id~ : Int,
  limits~ : WorkerProtocolLimits,
) -> CompletionEntry raise WorkerProtocolError {
  if token_id < 0 || token_id > limits.max_token_id() {
    raise InvalidCompletionToken(slot=slot.value())
  }
  { slot, request_id, request_generation, outcome: DecodeCompleted(token_id~) }
}

///|
pub fn CompletionEntry::failed(
  slot~ : CompletionSlot,
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  failure~ : WorkerFailure,
) -> CompletionEntry {
  { slot, request_id, request_generation, outcome: Failed(failure) }
}

///|
pub fn CompletionEntry::slot(self : CompletionEntry) -> CompletionSlot {
  self.slot
}

///|
pub fn CompletionEntry::request_id(
  self : CompletionEntry,
) -> @inference.RequestId {
  self.request_id
}

///|
pub fn CompletionEntry::request_generation(
  self : CompletionEntry,
) -> @inference.RequestGeneration {
  self.request_generation
}

///|
pub fn CompletionEntry::outcome(self : CompletionEntry) -> CompletionOutcome {
  self.outcome
}

///|
/// Returns whether this result still belongs to the scheduler's current
/// request generation. The scheduler calls this after retiring submitted work
/// and before publishing a token or reusing request-owned KV state.
pub fn CompletionEntry::is_current(
  self : CompletionEntry,
  request_id : @inference.RequestId,
  current_generation : @inference.RequestGeneration,
) -> Bool {
  self.request_id == request_id && self.request_generation == current_generation
}

///|
/// Immutable worker-to-scheduler result for one submitted plan. It deliberately
/// has no `Debug` implementation.
pub struct CompletionRecord {
  sequence : PlanSequence
  model_plan_generation : ModelPlanGeneration
  entries : ReadOnlyArray[CompletionEntry]
} derive(Eq)

///|
/// Defensively copies a bounded result table and rejects duplicate slots.
pub fn CompletionRecord::new(
  sequence~ : PlanSequence,
  model_plan_generation~ : ModelPlanGeneration,
  entries~ : ArrayView[CompletionEntry],
  limits~ : WorkerProtocolLimits,
) -> CompletionRecord raise WorkerProtocolError {
  if entries.length() == 0 || entries.length() > limits.max_plan_rows() {
    raise CompletionCountMismatch
  }
  for index, entry in entries {
    for previous in 0..<index {
      if entries[previous].slot() == entry.slot() {
        raise DuplicateCompletionSlot(index~)
      }
    }
    match entry.outcome() {
      PrefillCompleted(processed_tokens~) =>
        if processed_tokens <= 0 || processed_tokens > limits.max_plan_tokens() {
          raise CompletionTokenCountMismatch(index~)
        }
      DecodeCompleted(token_id~) =>
        if token_id < 0 || token_id > limits.max_token_id() {
          raise InvalidCompletionToken(slot=entry.slot().value())
        }
      Failed(_) => ()
    }
  }
  {
    sequence,
    model_plan_generation,
    entries: ReadOnlyArray::from_array(entries),
  }
}

///|
pub fn CompletionRecord::sequence(self : CompletionRecord) -> PlanSequence {
  self.sequence
}

///|
pub fn CompletionRecord::model_plan_generation(
  self : CompletionRecord,
) -> ModelPlanGeneration {
  self.model_plan_generation
}

///|
pub fn CompletionRecord::entries(
  self : CompletionRecord,
) -> ReadOnlyArray[CompletionEntry] {
  self.entries
}

///|
fn find_entry(
  entries : ReadOnlyArray[CompletionEntry],
  slot : CompletionSlot,
) -> CompletionEntry? {
  for entry in entries {
    if entry.slot() == slot {
      return Some(entry)
    }
  }
  None
}

///|
fn validate_completion_identity(
  entry : CompletionEntry,
  request_id : @inference.RequestId,
  request_generation : @inference.RequestGeneration,
  index : Int,
) -> Unit raise WorkerProtocolError {
  if entry.request_id() != request_id ||
    entry.request_generation() != request_generation {
    raise CompletionIdentityMismatch(index~)
  }
}

///|
fn validate_prefill_completion(
  entries : ReadOnlyArray[CompletionEntry],
  row : PrefillRow,
  index : Int,
) -> Unit raise WorkerProtocolError {
  guard find_entry(entries, row.completion_slot()) is Some(entry) else {
    raise UnknownCompletionSlot(index~)
  }
  validate_completion_identity(
    entry,
    row.request_id(),
    row.request_generation(),
    index,
  )
  match entry.outcome() {
    PrefillCompleted(processed_tokens~) =>
      if processed_tokens != row.tokens().length() {
        raise CompletionTokenCountMismatch(index~)
      }
    Failed(_) => ()
    DecodeCompleted(_) => raise CompletionKindMismatch(index~)
  }
}

///|
fn validate_decode_completion(
  entries : ReadOnlyArray[CompletionEntry],
  row : DecodeRow,
  index : Int,
) -> Unit raise WorkerProtocolError {
  guard find_entry(entries, row.completion_slot()) is Some(entry) else {
    raise UnknownCompletionSlot(index~)
  }
  validate_completion_identity(
    entry,
    row.request_id(),
    row.request_generation(),
    index,
  )
  match entry.outcome() {
    DecodeCompleted(_) | Failed(_) => ()
    PrefillCompleted(_) => raise CompletionKindMismatch(index~)
  }
}

///|
/// Authenticates a completion against the exact immutable work submitted to the
/// worker. This intentionally does not consult mutable scheduler state: stale
/// work must remain retireable. Before publishing output or reusing request KV,
/// the scheduler must separately call `CompletionEntry::is_current` with its
/// current request generation.
pub fn CompletionRecord::validate_for(
  self : CompletionRecord,
  submitted : SchedulePlan,
) -> Unit raise WorkerProtocolError {
  if self.sequence != submitted.sequence() {
    raise StalePlanSequence
  }
  if self.model_plan_generation != submitted.model_plan_generation() {
    raise StaleModelPlanGeneration
  }
  if self.entries.length() != submitted.row_count() {
    raise CompletionCountMismatch
  }
  for index, row in submitted.prefill_rows() {
    validate_prefill_completion(self.entries, row, index)
  }
  let prefill_count = submitted.prefill_rows().length()
  for index, row in submitted.decode_rows() {
    validate_decode_completion(self.entries, row, prefill_count + index)
  }
}
