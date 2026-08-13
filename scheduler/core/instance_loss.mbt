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
