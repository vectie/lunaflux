///|
fn Scheduler::side_slots(self : Scheduler, side : Int) -> FixedArray[Int] {
  if side == 0 {
    self.plan_a_slots
  } else {
    self.plan_b_slots
  }
}

///|
fn Scheduler::completion_entry_for_row(
  self : Scheduler,
  completion : @worker_protocol.SubmittedCompletion,
  row : Int,
) -> @worker_protocol.SubmittedCompletionEntry raise SchedulerError {
  try completion.entry_for_slot(self.completion_slots[row]) catch {
    error => raise WorkerProtocol(error)
  } noraise {
    entry => entry
  }
}

///|
fn Scheduler::row_is_current(
  self : Scheduler,
  slot : Int,
  entry : @worker_protocol.SubmittedCompletionEntry,
) -> Bool raise SchedulerError {
  if slot < 0 || slot >= self.slot_states.length() || !self.in_flight[slot] {
    raise invalid(RequestState, InvalidTransition)
  }
  let request = match self.requests[slot] {
    Some(value) => value
    None => raise invalid(RequestState, InvalidTransition)
  }
  let entry_id = try entry.request_id() catch {
    error => raise WorkerProtocol(error)
  } noraise {
    value => value
  }
  if entry_id != request.request_id() {
    raise invalid(Completion, Mismatch)
  }
  let entry_generation = try entry.request_generation() catch {
    error => raise WorkerProtocol(error)
  } noraise {
    value => value
  }
  if entry_generation == self.request_generations[slot] {
    if self.slot_states[slot] != SLOT_PREFILL &&
      self.slot_states[slot] != SLOT_DECODE {
      raise invalid(RequestState, InvalidTransition)
    }
    true
  } else {
    if self.slot_states[slot] != SLOT_CANCELLED &&
      self.slot_states[slot] != SLOT_DEADLINE {
      raise invalid(Completion, Stale)
    }
    false
  }
}

///|
fn request_stops_on_token(request : TokenizedRequest, token : Int) -> Bool {
  let stop_tokens = request.stops().token_ids()
  for index in 0..<stop_tokens.length() {
    if stop_tokens[index] == token {
      return true
    }
  }
  false
}

///|
fn Scheduler::validate_prefill_progress(
  self : Scheduler,
  plan : @worker_protocol.SubmittedSchedulePlan,
  row : Int,
  slot : Int,
  entry : @worker_protocol.SubmittedCompletionEntry,
) -> Unit raise SchedulerError {
  let descriptor = try plan.prefill_row(row) catch {
    error => raise WorkerProtocol(error)
  } noraise {
    value => value
  }
  let start = try descriptor.sequence_token_start() catch {
    error => raise WorkerProtocol(error)
  } noraise {
    value => value
  }
  let processed = try entry.processed_tokens() catch {
    error => raise WorkerProtocol(error)
  } noraise {
    value => value
  }
  let request = self.requests[slot].unwrap()
  if start != self.processed_inputs[slot] ||
    self.output_counts[slot] != 0 ||
    processed <= 0 ||
    processed > request.input_tokens().length() - start {
    raise invalid(Completion, Mismatch)
  }
  let produces_token = try descriptor.produces_token() catch {
    error => raise WorkerProtocol(error)
  } noraise {
    value => value
  }
  if produces_token != (start + processed == request.input_tokens().length()) {
    raise invalid(Completion, Mismatch)
  }
}

///|
fn Scheduler::validate_decode_progress(
  self : Scheduler,
  plan : @worker_protocol.SubmittedSchedulePlan,
  decode_row : Int,
  slot : Int,
) -> Unit raise SchedulerError {
  let descriptor = try plan.decode_row(decode_row) catch {
    error => raise WorkerProtocol(error)
  } noraise {
    value => value
  }
  let position = try descriptor.sequence_token_position() catch {
    error => raise WorkerProtocol(error)
  } noraise {
    value => value
  }
  let request = self.requests[slot].unwrap()
  if self.output_counts[slot] <= 0 ||
    self.output_counts[slot] >= request.effective_limits().max_new_tokens() ||
    position != request.input_tokens().length() + self.output_counts[slot] - 1 {
    raise invalid(Completion, Mismatch)
  }
}

///|
fn Scheduler::preflight_completion(
  self : Scheduler,
  plan : @worker_protocol.SubmittedSchedulePlan,
  completion : @worker_protocol.SubmittedCompletion,
  side : Int,
) -> Unit raise SchedulerError {
  let sequence = try plan.sequence_wire() catch {
    error => raise WorkerProtocol(error)
  } noraise {
    value => value
  }
  if self.last_retired_plan_sequence == MAX_GENERATION ||
    sequence != self.last_retired_plan_sequence + 1UL {
    raise invalid(Completion, OutOfOrder)
  }
  let plan_owner = if side == 0 { self.plan_a } else { self.plan_b }
  let completion_owner = if side == 0 {
    self.completion_a
  } else {
    self.completion_b
  }
  plan_owner.validate_submitted(plan) catch {
    error => raise WorkerProtocol(error)
  }
  completion_owner.validate_submitted(completion, plan) catch {
    error => raise WorkerProtocol(error)
  }
  if plan_owner.epoch() == MAX_GENERATION ||
    completion_owner.epoch() == MAX_GENERATION {
    raise invalid(Completion, Exhausted)
  }
  let row_count = try plan.row_count() catch {
    error => raise WorkerProtocol(error)
  } noraise {
    value => value
  }
  let prefill_count = try plan.prefill_count() catch {
    error => raise WorkerProtocol(error)
  } noraise {
    value => value
  }
  let slots = self.side_slots(side)
  let mut output_demand = 0
  let mut terminal_demand = 0
  for row in 0..<row_count {
    let slot = slots[row]
    let entry = self.completion_entry_for_row(completion, row)
    let current = self.row_is_current(slot, entry)
    let kind = try entry.kind() catch {
      error => raise WorkerProtocol(error)
    } noraise {
      value => value
    }
    if current &&
      (
        (row < prefill_count && self.slot_states[slot] != SLOT_PREFILL) ||
        (row >= prefill_count && self.slot_states[slot] != SLOT_DECODE)
      ) {
      raise invalid(RequestState, InvalidTransition)
    }
    if row < prefill_count && kind != FailedCompletion {
      self.validate_prefill_progress(plan, row, slot, entry)
    } else if row >= prefill_count && kind != FailedCompletion {
      self.validate_decode_progress(plan, row - prefill_count, slot)
    }
    let mut release = !current
    if current {
      match kind {
        PrefillCompletion => ()
        FinalPrefillCompletion | DecodeCompletion => {
          let token = try entry.token_id() catch {
            error => raise WorkerProtocol(error)
          } noraise {
            value => value
          }
          let request = self.requests[slot].unwrap()
          output_demand += 1
          if request_stops_on_token(request, token) ||
            self.output_counts[slot] + 1 >=
            request.effective_limits().max_new_tokens() {
            terminal_demand += 1
            release = true
          }
        }
        FailedCompletion => {
          terminal_demand += 1
          release = true
        }
      }
    }
    if release {
      self.preflight_active_release(slot)
    }
  }
  if output_demand > self.output_request_ids.length() - self.output_count {
    raise invalid(OutputCapacity, Backpressured)
  }
  if terminal_demand > self.terminal_available() {
    raise invalid(TerminalNoticeCapacity, Backpressured)
  }
  self.preflight_publications(output_demand + terminal_demand)
}
