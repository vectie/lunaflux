///|
fn CompletionWriter::owner(
  self : CompletionWriter,
) -> CompletionBuffer raise WorkerProtocolError {
  let completion_owner = self.0
  if self.1 != completion_owner.epoch {
    raise StaleCompletionBufferEpoch
  }
  completion_owner.require_state(CompletionWritable)
  if !completion_owner.writer_open {
    raise CompletionWriterNotOpen
  }
  ignore(SubmittedSchedulePlan(self.2, self.3).authenticate())
  completion_owner
}

///|
/// Issues a worker-only writer tied to this exact writable completion epoch
/// and submitted plan epoch. The scheduler retains both mutable owners.
pub fn CompletionBuffer::writer_for(
  self : CompletionBuffer,
  plan : SubmittedSchedulePlan,
) -> CompletionWriter raise WorkerProtocolError {
  self.require_state(CompletionWritable)
  if self.entry_count != 0 {
    raise CompletionCountMismatch
  }
  if self.writer_open {
    raise CompletionWriterAlreadyOpen
  }
  ignore(plan.authenticate())
  self.writer_open = true
  CompletionWriter(self, self.epoch, plan.0, plan.1)
}

///|
pub fn CompletionWriter::append_prefill_completed(
  self : CompletionWriter,
  slot~ : CompletionSlot,
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  processed_tokens~ : Int,
) -> Unit raise WorkerProtocolError {
  self
  .owner()
  .append_prefill_completed_unleased(
    slot~,
    request_id~,
    request_generation~,
    processed_tokens~,
  )
}

///|
pub fn CompletionWriter::append_final_prefill_completed(
  self : CompletionWriter,
  slot~ : CompletionSlot,
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  processed_tokens~ : Int,
  token_id~ : Int,
) -> Unit raise WorkerProtocolError {
  self
  .owner()
  .append_final_prefill_completed_unleased(
    slot~,
    request_id~,
    request_generation~,
    processed_tokens~,
    token_id~,
  )
}

///|
pub fn CompletionWriter::append_decode_completed(
  self : CompletionWriter,
  slot~ : CompletionSlot,
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  token_id~ : Int,
) -> Unit raise WorkerProtocolError {
  self
  .owner()
  .append_decode_completed_unleased(
    slot~,
    request_id~,
    request_generation~,
    token_id~,
  )
}

///|
pub fn CompletionWriter::append_failed(
  self : CompletionWriter,
  slot~ : CompletionSlot,
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  failure~ : WorkerFailure,
) -> Unit raise WorkerProtocolError {
  self
  .owner()
  .append_failed_unleased(slot~, request_id~, request_generation~, failure~)
}

///|
/// Aborts a partial writable epoch after worker construction or validation
/// fails. Advancing the completion epoch makes every outstanding writer stale;
/// the same still-submitted plan can then receive a fresh writer.
pub fn CompletionWriter::abort(
  self : CompletionWriter,
) -> Unit raise WorkerProtocolError {
  let owner = self.owner()
  if owner.epoch == 0xffffffffffffffffUL {
    raise IdentityOverflow(CompletionBufferEpochIdentity)
  }
  owner.entry_count = 0
  owner.writer_open = false
  owner.epoch = owner.epoch + 1UL
}

///|
/// Authenticates and freezes this exact completion epoch for the plan bound at
/// lease creation. No mutable owner or plan argument is exposed to the worker.
pub fn CompletionWriter::submit(
  self : CompletionWriter,
) -> SubmittedCompletion raise WorkerProtocolError {
  let owner = self.owner()
  let submitted = owner.submit_for_unleased(
    SubmittedSchedulePlan(self.2, self.3),
  )
  owner.writer_open = false
  submitted
}
