///|
/// Expires one exact request without scanning or mutating unrelated requests.
/// `None` means its absolute deadline has not elapsed.
pub fn Scheduler::expire_exact(
  self : Scheduler,
  handle : RequestHandle,
  now_millis : UInt64,
) -> @inference.RequestGeneration? raise SchedulerError {
  self.require_no_cancel_reservation()
  let index = self.checked_handle_index(handle)
  if self.slot_states[index] == SLOT_CANCELLED ||
    self.slot_states[index] == SLOT_DEADLINE ||
    self.slot_states[index] == SLOT_FAILED {
    raise invalid(RequestState, InvalidTransition)
  }
  if now_millis < self.deadlines[index] {
    return None
  }
  if self.terminal_available() == 0 {
    raise invalid(TerminalNoticeCapacity, Backpressured)
  }
  if self.generation_cursor == MAX_GENERATION {
    raise invalid(RequestGeneration, Exhausted)
  }
  self.preflight_publications(1)
  if self.slot_states[index] != SLOT_WAITING && !self.in_flight[index] {
    self.preflight_active_release(index)
  }
  if self.slot_states[index] == SLOT_WAITING {
    self.terminate_waiting(index, SLOT_DEADLINE, DeadlineReached)
    return Some(self.request_generations[index])
  }
  self.request_generations[index] = self.allocate_request_generation()
  self.slot_states[index] = SLOT_DEADLINE
  let request_id = match self.requests[index] {
    Some(value) => value.request_id()
    None => raise invalid(RequestState, InvalidTransition)
  }
  self.enqueue_terminal(
    index,
    request_id,
    self.request_generations[index],
    DeadlineReached,
  )
  if !self.in_flight[index] {
    self.release_active_request(index)
  }
  Some(self.request_generations[index])
}

///|
pub fn Scheduler::expire_deadlines(
  self : Scheduler,
  now_millis : UInt64,
) -> Int raise SchedulerError {
  self.require_no_cancel_reservation()
  if !self.deadline_root_due(now_millis) {
    self.deadline_live_visits = if self.deadline_heap_count == 0 {
      0
    } else {
      1
    }
    self.deadline_due_pops = 0
    return 0
  }
  let candidate_count = self.prepare_deadline_due_heap(now_millis)
  if candidate_count > self.terminal_available() {
    raise invalid(TerminalNoticeCapacity, Backpressured)
  }
  if self.generation_cursor > MAX_GENERATION - candidate_count.to_uint64() {
    raise invalid(RequestGeneration, Exhausted)
  }
  self.preflight_publications(candidate_count)
  for due_index in 0..<candidate_count {
    let slot = self.deadline_due_heap[due_index]
    if self.slot_states[slot] != SLOT_WAITING && !self.in_flight[slot] {
      self.preflight_active_release(slot)
    }
  }
  let mut expired = 0
  let mut due_left = candidate_count
  while due_left > 0 {
    let index = self.pop_deadline_due(due_left)
    due_left -= 1
    let state = self.slot_states[index]
    if state == SLOT_WAITING {
      self.terminate_waiting(index, SLOT_DEADLINE, DeadlineReached)
    } else {
      self.request_generations[index] = self.allocate_request_generation()
      self.slot_states[index] = SLOT_DEADLINE
      let request_id = match self.requests[index] {
        Some(value) => value.request_id()
        None => raise invalid(RequestState, InvalidTransition)
      }
      self.enqueue_terminal(
        index,
        request_id,
        self.request_generations[index],
        DeadlineReached,
      )
      if !self.in_flight[index] {
        self.release_active_request(index)
      }
    }
    expired += 1
  }
  expired
}
