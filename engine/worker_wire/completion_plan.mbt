///|
fn validate_completion_identity(
  entry : CompletionFrameEntry,
  slot : @worker_protocol.CompletionSlot,
  request_id : @inference.RequestId,
  request_generation : @inference.RequestGeneration,
) -> Unit raise WorkerWireError {
  if entry.slot() != slot.value() ||
    entry.request_id_value() != request_id.value() ||
    entry.request_generation_value() != request_generation.value() {
    raise InvalidFrame(CompletionEntry)
  }
}

///|
fn validate_prefill_completion(
  frame : ValidatedCompletionFrame,
  row : @worker_protocol.SubmittedPrefillRow,
) -> Unit raise WorkerWireError {
  let slot = row.completion_slot() catch { error => raise Protocol(error) }
  let request_id = row.request_id() catch { error => raise Protocol(error) }
  let generation = row.request_generation() catch {
    error => raise Protocol(error)
  }
  let tokens = row.tokens() catch { error => raise Protocol(error) }
  let produces_token = row.produces_token() catch {
    error => raise Protocol(error)
  }
  let entry = frame.entry_for_slot(slot.value())
  validate_completion_identity(entry, slot, request_id, generation)
  let kind = entry.kind()
  if kind == FailedCompletion {
    return
  }
  let expected_kind : @worker_protocol.CompletionEntryKind = if produces_token {
    FinalPrefillCompletion
  } else {
    PrefillCompletion
  }
  if kind != expected_kind || entry.processed_tokens() != tokens.length() {
    raise InvalidFrame(CompletionEntry)
  }
}

///|
fn validate_decode_completion(
  frame : ValidatedCompletionFrame,
  row : @worker_protocol.SubmittedDecodeRow,
) -> Unit raise WorkerWireError {
  let slot = row.completion_slot() catch { error => raise Protocol(error) }
  let request_id = row.request_id() catch { error => raise Protocol(error) }
  let generation = row.request_generation() catch {
    error => raise Protocol(error)
  }
  let entry = frame.entry_for_slot(slot.value())
  validate_completion_identity(entry, slot, request_id, generation)
  let kind = entry.kind()
  if kind != DecodeCompletion && kind != FailedCompletion {
    raise InvalidFrame(CompletionEntry)
  }
}

///|
/// Authenticates this untrusted completion frame against one exact submitted
/// plan. The frame must contain exactly one identity- and kind-correct outcome
/// for every plan row; worker failures are valid outcomes for either row kind.
pub fn ValidatedCompletionFrame::validate_for(
  self : ValidatedCompletionFrame,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> Unit raise WorkerWireError {
  let sequence = plan.sequence_wire() catch { error => raise Protocol(error) }
  let generation = plan.model_plan_generation_wire() catch {
    error => raise Protocol(error)
  }
  let prefill_count = plan.prefill_count() catch {
    error => raise Protocol(error)
  }
  let decode_count = plan.decode_count() catch {
    error => raise Protocol(error)
  }
  if self.sequence_value() != sequence ||
    self.model_generation_value() != generation ||
    self.entry_count() != prefill_count + decode_count {
    raise InvalidFrame(CompletionEntry)
  }
  for index in 0..<prefill_count {
    validate_prefill_completion(
      self,
      plan.prefill_row(index) catch {
        error => raise Protocol(error)
      },
    )
  }
  for index in 0..<decode_count {
    validate_decode_completion(
      self,
      plan.decode_row(index) catch {
        error => raise Protocol(error)
      },
    )
  }
}
