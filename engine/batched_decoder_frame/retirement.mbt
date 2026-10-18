///|
fn BatchedDecoderFrame::commit_slots(self : Self) -> Unit {
  for entry in 0..<self.active_sequences {
    let slot = self.pending_slots[entry]
    self.live[slot] = true
    self.ids[slot] = self.pending_ids[entry]
    self.generations[slot] = self.pending_generations[entry]
    self.processed[slot] = self.pending_ends[entry]
  }
  self.state = Idle
}

///|
/// Publish once, after every rank and device error port retires. Partial device
/// mutation cannot be undone by retrying the same wire plan: execution failure
/// poisons this owner. Invalid samples are failures, never successful tokens.
pub fn BatchedDecoderFrame::complete(
  self : Self,
  sampled_tokens : ArrayView[Int],
  execution_failed~ : Bool,
) -> @wire.ValidatedCompletionFrame raise {
  self.require_staged()
  guard execution_failed || sampled_tokens.length() == self.outputs else {
    raise BatchFrameError::InvalidCapacity
  }
  let failed = execution_failed ||
    sampled_tokens.any(token => token < 0 || token >= self.vocabulary)
  let frame = self.completion.plan()
  let writer = self.completion.writer()
  self.state = Failed
  let prefills = frame.prefill_count()
  for entry in 0..<prefills {
    let row = frame.prefill_row(entry)
    let output = self.entry_outputs[entry]
    if failed {
      writer.append_prefill_failed(row, ExecutionFailed)
    } else if output >= 0 {
      writer.append_final_prefill_completed(row, sampled_tokens[output])
    } else {
      writer.append_prefill_completed(row)
    }
  }
  for index in 0..<frame.decode_count() {
    let row = frame.decode_row(index)
    if failed {
      writer.append_decode_failed(row, ExecutionFailed)
    } else {
      writer.append_decode_completed(
        row,
        sampled_tokens[self.entry_outputs[prefills + index]],
      )
    }
  }
  let result = self.completion.submit()
  if !failed {
    self.commit_slots()
  }
  result
}

///|
/// A non-publishing TP/pipeline rank commits only on whole-step acceptance.
pub fn BatchedDecoderFrame::retire_stage(
  self : Self,
  execution_failed~ : Bool,
) -> Unit raise {
  self.require_staged()
  self.state = Failed
  self.completion.close()
  if !execution_failed {
    self.commit_slots()
  }
}

///|
/// Release only after the GPU and scheduler have retired the request. A late
/// cancellation for an old generation cannot release a reused live slot.
pub fn BatchedDecoderFrame::release_request(
  self : Self,
  request_id : UInt64,
  request_generation : UInt64,
) -> Unit raise BatchFrameError {
  self.require_idle()
  let slot = self.find_live(request_id)
  guard slot >= 0 else { raise UnknownRequest }
  guard self.generations[slot] == request_generation else {
    raise GenerationMismatch
  }
  self.live[slot] = false
  self.processed[slot] = 0
}

///|
pub fn BatchedDecoderFrame::processed_tokens(
  self : Self,
  request_id : UInt64,
  request_generation : UInt64,
) -> Int raise BatchFrameError {
  guard !(self.state is Closed) else { raise Closed }
  let slot = self.find_live(request_id)
  guard slot >= 0 else { raise UnknownRequest }
  guard self.generations[slot] == request_generation else {
    raise GenerationMismatch
  }
  self.processed[slot]
}

///|
pub fn BatchedDecoderFrame::completion(
  self : Self,
) -> @wire.ValidatedCompletionFrame raise {
  guard !(self.state is Closed) else { raise BatchFrameError::Closed }
  self.completion.completion()
}

///|
pub fn BatchedDecoderFrame::is_faulted(self : Self) -> Bool {
  self.state is Failed
}

///|
/// The executor must drain its borrowed ports before the frame is closed.
pub fn BatchedDecoderFrame::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.completion.close()
  self.state = Closed
}
