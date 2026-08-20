///|
fn Scheduler::enqueue_generated_token(
  self : Scheduler,
  slot : Int,
  token : Int,
) -> Unit raise SchedulerError {
  let request = self.requests[slot].unwrap()
  let sequence = self.allocate_publication_sequence()
  let index = self.output_tail
  self.output_request_ids[index] = request.request_id()
  self.output_generations[index] = self.request_generations[slot]
  self.output_token_ids[index] = token
  self.output_positions[index] = self.output_counts[slot]
  self.output_sequences[index] = sequence
  self.output_tail = (index + 1) % self.output_request_ids.length()
  self.output_count += 1
}

///|
fn Scheduler::terminal_reason_for_token(
  self : Scheduler,
  slot : Int,
  token : Int,
) -> TerminalReason {
  if request_stops_on_token(self.requests[slot].unwrap(), token) {
    StopTokenReached
  } else {
    MaximumOutputReached
  }
}

///|
fn Scheduler::commit_current_completion(
  self : Scheduler,
  slot : Int,
  entry : @worker_protocol.SubmittedCompletionEntry,
) -> Unit {
  let kind = try! entry.kind()
  match kind {
    PrefillCompletion => {
      self.processed_inputs[slot] += try! entry.processed_tokens()
      self.slot_states[slot] = SLOT_PREFILL
      self.in_flight[slot] = false
    }
    FinalPrefillCompletion => {
      let token = try! entry.token_id()
      self.processed_inputs[slot] += try! entry.processed_tokens()
      try! self.enqueue_generated_token(slot, token)
      self.decode_tokens[slot] = token
      self.output_counts[slot] += 1
      let request = self.requests[slot].unwrap()
      if request_stops_on_token(request, token) ||
        self.output_counts[slot] >= request.effective_limits().max_new_tokens() {
        try! self.enqueue_terminal(
          request.request_id(),
          self.request_generations[slot],
          self.terminal_reason_for_token(slot, token),
        )
        self.slot_states[slot] = SLOT_FINISHING
        self.release_active_request(slot)
      } else {
        self.slot_states[slot] = SLOT_DECODE
        self.in_flight[slot] = false
      }
    }
    DecodeCompletion => {
      let token = try! entry.token_id()
      try! self.enqueue_generated_token(slot, token)
      self.decode_tokens[slot] = token
      self.output_counts[slot] += 1
      let request = self.requests[slot].unwrap()
      if request_stops_on_token(request, token) ||
        self.output_counts[slot] >= request.effective_limits().max_new_tokens() {
        try! self.enqueue_terminal(
          request.request_id(),
          self.request_generations[slot],
          self.terminal_reason_for_token(slot, token),
        )
        self.slot_states[slot] = SLOT_FINISHING
        self.release_active_request(slot)
      } else {
        self.slot_states[slot] = SLOT_DECODE
        self.in_flight[slot] = false
      }
    }
    FailedCompletion => {
      let request = self.requests[slot].unwrap()
      try! self.enqueue_terminal(
        request.request_id(),
        self.request_generations[slot],
        WorkerFailed,
      )
      self.slot_states[slot] = SLOT_FAILED
      self.release_active_request(slot)
    }
  }
}

///|
fn Scheduler::finish_protocol_retirement(
  self : Scheduler,
  plan : @worker_protocol.SubmittedSchedulePlan,
  completion : @worker_protocol.SubmittedCompletion,
  side : Int,
  row_count : Int,
  sequence : UInt64,
) -> Unit {
  let plan_owner = if side == 0 { self.plan_a } else { self.plan_b }
  let completion_owner = if side == 0 {
    self.completion_a
  } else {
    self.completion_b
  }
  let consumed = try! completion_owner.consume(completion, plan)
  let retired = try! plan_owner.retire(plan)
  try! completion_owner.reset(consumed)
  try! plan_owner.reset(retired)
  let slots = self.side_slots(side)
  for row in 0..<row_count {
    slots[row] = NO_INDEX
  }
  if side == 0 {
    self.plan_a_sequence = 0UL
  } else {
    self.plan_b_sequence = 0UL
  }
  self.last_retired_plan_sequence = sequence
}

///|
/// Authenticates and commits one complete worker result in plan-row order.
/// Every output, terminal, and resource obligation is preflighted before any
/// request or protocol lifecycle state changes.
pub fn Scheduler::complete(
  self : Scheduler,
  plan : @worker_protocol.SubmittedSchedulePlan,
  completion : @worker_protocol.SubmittedCompletion,
) -> CompletionCommit raise SchedulerError {
  self.require_no_cancel_reservation()
  let side = self.completion_side(plan)
  match self.preflight_completion(plan, completion, side) {
    CompletionCommitted => ()
    CompletionBackpressured => return CompletionBackpressured
  }
  let sequence = try! plan.sequence_wire()
  let row_count = try! plan.row_count()
  let slots = self.side_slots(side)
  for row in 0..<row_count {
    let slot = slots[row]
    let entry = try! self.completion_entry_for_row(completion, row)
    let current = try! self.row_is_current(slot, entry)
    if current {
      self.commit_current_completion(slot, entry)
    } else {
      self.release_active_request(slot)
    }
  }
  self.finish_protocol_retirement(plan, completion, side, row_count, sequence)
  CompletionCommitted
}

///|
/// Retries the exact currently submitted completion paired with `plan`.
/// Transport owners need retain only their own flight state; the scheduler
/// reauthenticates its completion owner and epoch before normal retirement.
pub fn Scheduler::complete_submitted(
  self : Scheduler,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> CompletionCommit raise SchedulerError {
  self.require_no_cancel_reservation()
  let side = self.completion_side(plan)
  let owner = if side == 0 { self.completion_a } else { self.completion_b }
  let completion = owner.current_submitted(plan) catch {
    error => raise WorkerProtocol(error)
  }
  self.complete(plan, completion)
}
