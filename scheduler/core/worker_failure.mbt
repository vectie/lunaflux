///|
fn append_worker_failure_rows(
  writer : @worker_protocol.CompletionWriter,
  plan : @worker_protocol.SubmittedSchedulePlan,
  failure : @worker_protocol.WorkerFailure,
) -> Unit raise SchedulerError {
  let prefill_count = plan.prefill_count() catch {
    error => raise WorkerProtocol(error)
  }
  for index in 0..<prefill_count {
    let row = plan.prefill_row(index) catch {
      error => raise WorkerProtocol(error)
    }
    writer.append_failed(
      slot=row.completion_slot(),
      request_id=row.request_id(),
      request_generation=row.request_generation(),
      failure~,
    ) catch {
      error => raise WorkerProtocol(error)
    }
  }
  let decode_count = plan.decode_count() catch {
    error => raise WorkerProtocol(error)
  }
  for index in 0..<decode_count {
    let row = plan.decode_row(index) catch {
      error => raise WorkerProtocol(error)
    }
    writer.append_failed(
      slot=row.completion_slot(),
      request_id=row.request_id(),
      request_generation=row.request_generation(),
      failure~,
    ) catch {
      error => raise WorkerProtocol(error)
    }
  }
}

///|
fn[T] abort_failure_writer(
  writer : @worker_protocol.CompletionWriter,
  error : SchedulerError,
) -> T raise SchedulerError {
  try! writer.abort()
  raise error
}

///|
fn Scheduler::preflight_exhausted_failure_retirement(
  self : Scheduler,
  plan : @worker_protocol.SubmittedSchedulePlan,
  side : Int,
) -> Int raise SchedulerError {
  let plan_owner = if side == 0 { self.plan_a } else { self.plan_b }
  plan_owner.validate_submitted(plan) catch {
    error => raise WorkerProtocol(error)
  }
  let sequence = plan.sequence_wire() catch {
    error => raise WorkerProtocol(error)
  }
  if self.last_retired_plan_sequence == MAX_GENERATION ||
    sequence != self.last_retired_plan_sequence + 1UL {
    raise invalid(Completion, OutOfOrder)
  }
  let row_count = plan.row_count() catch {
    error => raise WorkerProtocol(error)
  }
  let prefill_count = plan.prefill_count() catch {
    error => raise WorkerProtocol(error)
  }
  let slots = self.side_slots(side)
  let mut terminal_demand = 0
  for row in 0..<row_count {
    let slot = slots[row]
    if slot < 0 || slot >= self.slot_states.length() || !self.in_flight[slot] {
      raise invalid(RequestState, InvalidTransition)
    }
    let request = match self.requests[slot] {
      Some(value) => value
      None => raise invalid(RequestState, InvalidTransition)
    }
    let (row_id, row_generation) = if row < prefill_count {
      let descriptor = plan.prefill_row(row) catch {
        error => raise WorkerProtocol(error)
      }
      (
        descriptor.request_id() catch {
          error => raise WorkerProtocol(error)
        },
        descriptor.request_generation() catch {
          error => raise WorkerProtocol(error)
        },
      )
    } else {
      let descriptor = plan.decode_row(row - prefill_count) catch {
        error => raise WorkerProtocol(error)
      }
      (
        descriptor.request_id() catch {
          error => raise WorkerProtocol(error)
        },
        descriptor.request_generation() catch {
          error => raise WorkerProtocol(error)
        },
      )
    }
    if row_id != request.request_id() {
      raise invalid(Completion, Mismatch)
    }
    let current_generation = self.request_generations[slot]
    if self.slot_states[slot] == SLOT_PREFILL ||
      self.slot_states[slot] == SLOT_DECODE {
      if row_generation != current_generation {
        raise invalid(Completion, Stale)
      }
    } else if self.slot_states[slot] == SLOT_CANCELLED ||
      self.slot_states[slot] == SLOT_DEADLINE {
      if row_generation.value() >= current_generation.value() {
        raise invalid(Completion, Mismatch)
      }
    } else {
      raise invalid(RequestState, InvalidTransition)
    }
    self.preflight_active_release(slot)
    if self.slot_states[slot] == SLOT_PREFILL ||
      self.slot_states[slot] == SLOT_DECODE {
      terminal_demand += 1
    } else if self.slot_states[slot] != SLOT_CANCELLED &&
      self.slot_states[slot] != SLOT_DEADLINE {
      raise invalid(RequestState, InvalidTransition)
    }
  }
  if terminal_demand > self.terminal_available() {
    raise invalid(TerminalNoticeCapacity, Backpressured)
  }
  self.preflight_publications(terminal_demand)
  row_count
}

///|
fn Scheduler::commit_exhausted_failure_retirement(
  self : Scheduler,
  plan : @worker_protocol.SubmittedSchedulePlan,
  side : Int,
  row_count : Int,
) -> Unit {
  let plan_owner = if side == 0 { self.plan_a } else { self.plan_b }
  let completion_owner = if side == 0 {
    self.completion_a
  } else {
    self.completion_b
  }
  let sequence = try! plan.sequence_wire()
  let invalid = try! completion_owner.current_submitted(plan)
  ignore(try! completion_owner.consume(invalid, plan))
  let slots = self.side_slots(side)
  for row in 0..<row_count {
    let slot = slots[row]
    if self.slot_states[slot] == SLOT_PREFILL ||
      self.slot_states[slot] == SLOT_DECODE {
      let request = self.requests[slot].unwrap()
      try! self.enqueue_terminal(
        slot,
        request.request_id(),
        self.request_generations[slot],
        WorkerFailed,
      )
      self.slot_states[slot] = SLOT_FAILED
    }
    self.release_active_request(slot)
    slots[row] = NO_INDEX
  }
  let retired = try! plan_owner.retire(plan)
  ignore(retired)
  if side == 0 {
    self.plan_a_sequence = 0UL
  } else {
    self.plan_b_sequence = 0UL
  }
  self.last_retired_plan_sequence = sequence
}

///|
fn requires_terminal_failure_replacement(
  completion_epoch : UInt64,
  plan_epoch : UInt64,
) -> Bool {
  completion_epoch >= MAX_GENERATION - 1UL || plan_epoch == MAX_GENERATION
}

///|
/// Commits one exact submitted plan as a bounded worker failure. If scheduler
/// publication is backpressured, the completed failure epoch remains submitted
/// and the same call reuses it on retry without reopening a writer.
pub fn Scheduler::fail_plan(
  self : Scheduler,
  plan : @worker_protocol.SubmittedSchedulePlan,
  failure : @worker_protocol.WorkerFailure,
) -> Unit raise SchedulerError {
  self.require_no_cancel_reservation()
  let side = self.completion_side(plan)
  let owner = if side == 0 { self.completion_a } else { self.completion_b }
  let completion = match owner.state() {
    CompletionWritable => {
      let writer = self.completion_writer(plan)
      append_worker_failure_rows(writer, plan, failure) catch {
        error => abort_failure_writer(writer, error)
      }
      writer.submit() catch {
        error => raise WorkerProtocol(error)
      }
    }
    CompletionSubmitted =>
      owner.current_submitted(plan) catch {
        error => raise WorkerProtocol(error)
      }
    Consumed => raise invalid(Completion, Stale)
  }
  match self.complete(plan, completion) {
    CompletionCommitted => ()
    CompletionBackpressured =>
      raise invalid(TerminalNoticeCapacity, Backpressured)
  }
}

///|
/// Replaces a semantically rejected, already-submitted completion with the
/// exact bounded worker-failure completion for the same still-live plan. The
/// Reusable owners are reset before `fail_plan`. If either paired protocol
/// epoch cannot safely complete replacement, both owners are terminally
/// consumed/retired after exact request failure publication; the other pair
/// remains independently reusable.
pub fn Scheduler::replace_submitted_completion_with_failure(
  self : Scheduler,
  plan : @worker_protocol.SubmittedSchedulePlan,
  failure : @worker_protocol.WorkerFailure,
) -> Unit raise SchedulerError {
  self.require_no_cancel_reservation()
  let side = self.completion_side(plan)
  let owner = if side == 0 { self.completion_a } else { self.completion_b }
  if owner.state() == CompletionSubmitted {
    let invalid = owner.current_submitted(plan) catch {
      error => raise WorkerProtocol(error)
    }
    let plan_owner = if side == 0 { self.plan_a } else { self.plan_b }
    if requires_terminal_failure_replacement(owner.epoch(), plan_owner.epoch()) {
      let row_count = self.preflight_exhausted_failure_retirement(plan, side)
      self.commit_exhausted_failure_retirement(plan, side, row_count)
      return
    }
    let consumed = owner.consume(invalid, plan) catch {
      error => raise WorkerProtocol(error)
    }
    owner.reset(consumed) catch {
      error => raise WorkerProtocol(error)
    }
  }
  self.fail_plan(plan, failure)
}
