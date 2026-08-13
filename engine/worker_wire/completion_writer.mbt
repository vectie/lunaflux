///|
fn completion_failure_value(failure : @worker_protocol.WorkerFailure) -> UInt {
  match failure {
    ExecutionFailed => 0
    ResourceUnavailable => 1
    CancelledBeforeExecution => 2
  }
}

///|
fn CompletionFrameWriter::owner(
  self : CompletionFrameWriter,
) -> CompletionFrameBuffer raise WorkerWireError {
  let owner = self.0
  if !owner.writer_open {
    raise CompletionFrameWriterNotOpen
  }
  if owner.epoch != self.1 || owner.length != 0 {
    raise StaleFrame
  }
  ignore(ValidatedPlanFrame(self.2, self.3).owner())
  owner
}

///|
/// Proves that this exclusive completion writer is bound to the exact
/// validated plan-frame owner and epoch supplied by device execution.
pub fn CompletionFrameWriter::validate_plan(
  self : CompletionFrameWriter,
  plan : ValidatedPlanFrame,
) -> Unit raise WorkerWireError {
  ignore(self.owner())
  ignore(plan.owner())
  if !physical_equal(self.2, plan.0) || self.3 != plan.1 {
    raise CompletionFrameWriterPlanMismatch
  }
}

///|
fn CompletionFrameWriter::validate_prefill_row(
  self : CompletionFrameWriter,
  row : PlanFramePrefillRow,
) -> CompletionFrameBuffer raise WorkerWireError {
  let owner = self.owner()
  if !physical_equal(row.0, self.2) || row.1 != self.3 {
    raise CompletionFrameWriterPlanMismatch
  }
  if owner.entry_count != row.2 {
    raise CompletionFrameWriterOrderMismatch
  }
  ignore(row.offset())
  owner
}

///|
fn CompletionFrameWriter::validate_decode_row(
  self : CompletionFrameWriter,
  row : PlanFrameDecodeRow,
) -> CompletionFrameBuffer raise WorkerWireError {
  let owner = self.owner()
  if !physical_equal(row.0, self.2) || row.1 != self.3 {
    raise CompletionFrameWriterPlanMismatch
  }
  let prefill_count = ValidatedPlanFrame(self.2, self.3).prefill_count()
  if owner.entry_count != prefill_count + row.2 {
    raise CompletionFrameWriterOrderMismatch
  }
  ignore(row.offset())
  owner
}

///|
fn write_direct_completion_entry(
  owner : CompletionFrameBuffer,
  slot : Int,
  request_id : UInt64,
  request_generation : UInt64,
  kind : UInt,
  processed_tokens : Int,
  token_id : Int,
  failure : UInt,
) -> Unit raise WorkerWireError {
  if owner.entry_count >= owner.worker_limits.max_completion_slots() ||
    slot < 0 ||
    slot >= owner.slot_to_entry.length() ||
    owner.slot_to_entry[slot] != -1 {
    raise InvalidFrame(CompletionSlot)
  }
  let index = owner.entry_count
  let offset = completion_entry_offset(index)
  write_i32(owner.storage, offset, slot)
  write_u32(owner.storage, offset + 4, kind)
  write_u64(owner.storage, offset + 8, request_id)
  write_u64(owner.storage, offset + 16, request_generation)
  write_i32(owner.storage, offset + 24, processed_tokens)
  write_i32(owner.storage, offset + 28, token_id)
  write_u32(owner.storage, offset + 32, failure)
  write_u32(owner.storage, offset + 36, 0)
  owner.slot_to_entry[slot] = index
  owner.entry_count = index + 1
}

///|
fn CompletionFrameWriter::validate_token(
  self : CompletionFrameWriter,
  token_id : Int,
) -> Unit raise WorkerWireError {
  let owner = self.owner()
  let max_token = if owner.worker_limits.max_token_id() <
    owner.inference_limits.max_token_id() {
    owner.worker_limits.max_token_id()
  } else {
    owner.inference_limits.max_token_id()
  }
  if token_id < 0 || token_id > max_token {
    raise InvalidFrame(CompletionEntry)
  }
}

///|
/// Opens exclusive completion-frame construction for one exact received plan.
/// Opening invalidates the prior completion-frame epoch before any bytes are
/// mutated; submit publishes the new epoch, while abort makes it permanently
/// stale and leaves the owner reusable.
pub fn CompletionFrameBuffer::writer_for(
  self : CompletionFrameBuffer,
  plan : ValidatedPlanFrame,
) -> CompletionFrameWriter raise WorkerWireError {
  self.require_no_writer()
  let plan_owner = plan.owner()
  if plan_owner.worker_limits != self.worker_limits ||
    plan_owner.inference_limits != self.inference_limits {
    raise CompletionFrameWriterPlanMismatch
  }
  let prefill_count = plan.prefill_count()
  let decode_count = plan.decode_count()
  let count = prefill_count + decode_count
  if count <= 0 || count > self.worker_limits.max_completion_slots() {
    raise InvalidFrame(Counts)
  }
  // Reserve one additional epoch so abort can always stale a partial writer.
  if self.epoch >= 0xfffffffffffffffeUL {
    raise FrameEpochOverflow
  }
  let length = exact_completion_frame_bytes(count)
  let sequence = plan.sequence_value()
  let generation = plan.model_generation_value()
  self.clear_slot_index()
  self.epoch += 1UL
  self.length = 0
  self.entry_count = 0
  self.writer_open = true
  write_completion_header(self.storage, length, sequence, generation, count)
  CompletionFrameWriter(self, self.epoch, plan_owner, plan.1, count)
}

///|
pub fn CompletionFrameWriter::append_prefill_completed(
  self : CompletionFrameWriter,
  row : PlanFramePrefillRow,
) -> Unit raise WorkerWireError {
  let owner = self.validate_prefill_row(row)
  if row.produces_token() {
    raise InvalidFrame(CompletionEntry)
  }
  write_direct_completion_entry(
    owner,
    row.completion_slot(),
    row.request_id_value(),
    row.request_generation_value(),
    0,
    row.token_length(),
    0,
    0,
  )
}

///|
pub fn CompletionFrameWriter::append_final_prefill_completed(
  self : CompletionFrameWriter,
  row : PlanFramePrefillRow,
  token_id : Int,
) -> Unit raise WorkerWireError {
  let owner = self.validate_prefill_row(row)
  if !row.produces_token() {
    raise InvalidFrame(CompletionEntry)
  }
  self.validate_token(token_id)
  write_direct_completion_entry(
    owner,
    row.completion_slot(),
    row.request_id_value(),
    row.request_generation_value(),
    1,
    row.token_length(),
    token_id,
    0,
  )
}

///|
pub fn CompletionFrameWriter::append_decode_completed(
  self : CompletionFrameWriter,
  row : PlanFrameDecodeRow,
  token_id : Int,
) -> Unit raise WorkerWireError {
  let owner = self.validate_decode_row(row)
  self.validate_token(token_id)
  write_direct_completion_entry(
    owner,
    row.completion_slot(),
    row.request_id_value(),
    row.request_generation_value(),
    2,
    0,
    token_id,
    0,
  )
}

///|
pub fn CompletionFrameWriter::append_prefill_failed(
  self : CompletionFrameWriter,
  row : PlanFramePrefillRow,
  failure : @worker_protocol.WorkerFailure,
) -> Unit raise WorkerWireError {
  let owner = self.validate_prefill_row(row)
  write_direct_completion_entry(
    owner,
    row.completion_slot(),
    row.request_id_value(),
    row.request_generation_value(),
    3,
    0,
    0,
    completion_failure_value(failure),
  )
}

///|
pub fn CompletionFrameWriter::append_decode_failed(
  self : CompletionFrameWriter,
  row : PlanFrameDecodeRow,
  failure : @worker_protocol.WorkerFailure,
) -> Unit raise WorkerWireError {
  let owner = self.validate_decode_row(row)
  write_direct_completion_entry(
    owner,
    row.completion_slot(),
    row.request_id_value(),
    row.request_generation_value(),
    3,
    0,
    0,
    completion_failure_value(failure),
  )
}

///|
/// Publishes the exact complete canonical frame and closes this writer lease.
pub fn CompletionFrameWriter::submit(
  self : CompletionFrameWriter,
) -> ValidatedCompletionFrame raise WorkerWireError {
  let owner = self.owner()
  if owner.entry_count != self.4 {
    raise InvalidFrame(Counts)
  }
  let length = exact_completion_frame_bytes(self.4)
  write_u32(
    owner.storage,
    completion_checksum_offset,
    completion_frame_checksum(owner.storage, length),
  )
  owner.length = length
  owner.writer_open = false
  ValidatedCompletionFrame(owner, owner.epoch)
}

///|
/// Aborts partial construction, stales this exact lease, and returns the frame
/// owner to an empty reusable state.
pub fn CompletionFrameWriter::abort(
  self : CompletionFrameWriter,
) -> Unit raise WorkerWireError {
  let owner = self.owner()
  owner.clear_slot_index()
  owner.entry_count = 0
  owner.length = 0
  owner.writer_open = false
  owner.epoch += 1UL
}
