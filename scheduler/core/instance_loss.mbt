///|
/// Retires at most one live request after an unrecoverable worker-instance
/// loss. Every request publishes `WorkerFailed`; active requests release exact
/// KV/page/table authority, while waiting requests are removed from FIFO.
/// Backpressure and generation exhaustion mutate nothing. `Remaining` reports
/// the live request count after this transition.
pub fn Scheduler::drain_instance_loss(
  self : Scheduler,
) -> InstanceLossDrain raise SchedulerError {
  self.require_no_cancel_reservation()
  if self.plan_a_sequence != 0UL ||
    self.plan_b_sequence != 0UL ||
    self.last_retired_plan_sequence != self.last_plan_sequence {
    raise invalid(Plan, InvalidTransition)
  }
  if self.live_count == 0 {
    return Complete
  }
  if self.terminal_available() == 0 {
    raise invalid(TerminalNoticeCapacity, Backpressured)
  }
  self.preflight_publications(1)
  if self.generation_cursor == MAX_GENERATION {
    raise invalid(RequestGeneration, Exhausted)
  }
  let slot = if self.active_count > 0 {
    self.active_slots[0]
  } else {
    self.waiting_head
  }
  if slot < 0 || slot >= self.slot_states.length() || self.in_flight[slot] {
    raise invalid(RequestState, InvalidTransition)
  }
  if self.slot_states[slot] == SLOT_WAITING {
    self.terminate_waiting(slot, SLOT_FAILED, WorkerFailed)
  } else {
    self.preflight_active_release(slot)
    let request = match self.requests[slot] {
      Some(value) => value
      None => raise invalid(RequestState, InvalidTransition)
    }
    self.request_generations[slot] = self.allocate_request_generation()
    self.slot_states[slot] = SLOT_FAILED
    self.enqueue_terminal(
      request.request_id(),
      self.request_generations[slot],
      WorkerFailed,
    )
    self.release_active_request(slot)
  }
  if self.live_count == 0 {
    Complete
  } else {
    Remaining(self.live_count)
  }
}

///|
/// Drains the exact authenticated request selected by the deterministic
/// one-request instance-loss policy and returns its fresh terminal generation.
pub fn Scheduler::drain_instance_loss_exact(
  self : Scheduler,
  handle : RequestHandle,
) -> ExactInstanceLossDrain raise SchedulerError {
  for offset in 0..<self.terminal_count {
    let notice_index = (self.terminal_head + offset) %
      self.terminal_request_ids.length()
    if self.terminal_request_ids[notice_index] == handle.request_id() &&
      self.terminal_generations[notice_index] == handle.request_generation() {
      if self.live_count != 0 {
        raise invalid(RequestCapacity, Mismatch)
      }
      let progress = self.drain_instance_loss()
      let generation = self.terminal_generations[notice_index]
      return match progress {
        Remaining(count) => ExactLossRemaining(count, generation)
        Complete => ExactLossComplete(generation)
      }
    }
  }
  if self.live_count != 1 {
    raise invalid(RequestCapacity, Mismatch)
  }
  let index = self.checked_handle_index(handle)
  let selected = if self.active_count > 0 {
    self.active_slots[0]
  } else {
    self.waiting_head
  }
  if selected != index {
    raise invalid(Request, Mismatch)
  }
  let progress = self.drain_instance_loss()
  let expected_generation = @inference.RequestGeneration::from_wire(
    self.generation_cursor,
  )
  for offset in 0..<self.terminal_count {
    let notice_index = (self.terminal_head + offset) %
      self.terminal_request_ids.length()
    if self.terminal_request_ids[notice_index] == handle.request_id() &&
      self.terminal_generations[notice_index] == expected_generation &&
      self.terminal_reasons[notice_index] == WorkerFailed {
      let generation = self.terminal_generations[notice_index]
      return match progress {
        Remaining(count) => ExactLossRemaining(count, generation)
        Complete => ExactLossComplete(generation)
      }
    }
  }
  abort("exact instance-loss terminal invariant")
}
