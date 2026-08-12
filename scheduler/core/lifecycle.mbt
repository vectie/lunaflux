///|
fn Scheduler::status_at(self : Scheduler, index : Int) -> RequestStatus {
  let request = match self.requests[index] {
    Some(value) => value
    None => abort("scheduler status invariant")
  }
  {
    handle: {
      slot: SchedulerSlotId(index, self.slot_generations[index]),
      request_id: request.request_id(),
      request_generation: self.request_generations[index],
    },
    state: scheduler_state(self.slot_states[index]),
    input_tokens: request.input_tokens().length(),
    processed_input_tokens: self.processed_inputs[index],
    output_tokens: self.output_counts[index],
    deadline_at_millis: self.deadlines[index],
    in_flight: self.in_flight[index],
  }
}

///|
pub fn Scheduler::status(
  self : Scheduler,
  request_id : @inference.RequestId,
) -> RequestStatus? {
  let index = self.find_request(request_id)
  if index == NO_INDEX {
    None
  } else {
    Some(self.status_at(index))
  }
}

///|
pub fn Scheduler::live_request_count(self : Scheduler) -> Int {
  self.live_count
}

///|
pub fn Scheduler::waiting_request_count(self : Scheduler) -> Int {
  self.waiting_count
}

///|
pub fn Scheduler::free_page_count(self : Scheduler) -> Int {
  self.page_allocator.free_page_count()
}

///|
pub fn Scheduler::allocated_block_table_count(self : Scheduler) -> Int {
  self.block_tables.allocated_table_count()
}

///|
pub fn Scheduler::model_plan_generation(
  self : Scheduler,
) -> @worker_protocol.ModelPlanGeneration {
  self.model_plan_generation
}

///|
pub fn Scheduler::retired_slot_count(self : Scheduler) -> Int {
  self.retired_slot_count
}

///|
pub fn Scheduler::terminal_notice_count(self : Scheduler) -> Int {
  self.terminal_count
}

///|
pub fn Scheduler::poll_terminal(self : Scheduler) -> TerminalNotice? {
  if self.terminal_count == 0 {
    return None
  }
  let index = self.terminal_head
  let notice = TerminalNotice(
    self.terminal_request_ids[index],
    self.terminal_generations[index],
    self.terminal_reasons[index],
  )
  self.terminal_head = (index + 1) % self.terminal_request_ids.length()
  self.terminal_count -= 1
  Some(notice)
}

///|
pub fn Scheduler::advance_step(self : Scheduler) -> Unit raise SchedulerError {
  if self.step == MAX_GENERATION {
    raise invalid(Plan, Exhausted)
  }
  self.step += 1UL
}

///|
fn Scheduler::terminate_waiting(
  self : Scheduler,
  index : Int,
  state : Int,
  reason : TerminalReason,
) -> Unit raise SchedulerError {
  let request_id = match self.requests[index] {
    Some(value) => value.request_id()
    None => raise invalid(RequestState, InvalidTransition)
  }
  self.remove_waiting(index)
  self.slot_states[index] = state
  self.request_generations[index] = self.allocate_request_generation()
  self.enqueue_terminal(request_id, self.request_generations[index], reason)
  self.recycle_slot(index)
}

///|
pub fn Scheduler::cancel(
  self : Scheduler,
  handle : RequestHandle,
) -> Unit raise SchedulerError {
  let index = self.checked_handle_index(handle)
  if self.slot_states[index] == SLOT_CANCELLED ||
    self.slot_states[index] == SLOT_DEADLINE ||
    self.slot_states[index] == SLOT_FAILED {
    raise invalid(RequestState, InvalidTransition)
  }
  if self.terminal_available() == 0 {
    raise invalid(TerminalNoticeCapacity, Backpressured)
  }
  if self.generation_cursor == MAX_GENERATION {
    raise invalid(RequestGeneration, Exhausted)
  }
  if self.slot_states[index] == SLOT_WAITING {
    self.terminate_waiting(index, SLOT_CANCELLED, CancelledByCaller)
    return
  }
  self.request_generations[index] = self.allocate_request_generation()
  self.slot_states[index] = SLOT_CANCELLED
  let request_id = match self.requests[index] {
    Some(value) => value.request_id()
    None => raise invalid(RequestState, InvalidTransition)
  }
  self.enqueue_terminal(
    request_id,
    self.request_generations[index],
    CancelledByCaller,
  )
}

///|
pub fn Scheduler::expire_deadlines(
  self : Scheduler,
  now_millis : UInt64,
) -> Int raise SchedulerError {
  let mut candidate_count = 0
  for index in 0..<self.slot_states.length() {
    let state = self.slot_states[index]
    if state != SLOT_FREE &&
      state != SLOT_RETIRED &&
      state != SLOT_CANCELLED &&
      state != SLOT_DEADLINE &&
      state != SLOT_FAILED &&
      now_millis >= self.deadlines[index] {
      candidate_count += 1
    }
  }
  if candidate_count > self.terminal_available() {
    raise invalid(TerminalNoticeCapacity, Backpressured)
  }
  if self.generation_cursor > MAX_GENERATION - candidate_count.to_uint64() {
    raise invalid(RequestGeneration, Exhausted)
  }
  let mut expired = 0
  for index in 0..<self.slot_states.length() {
    let state = self.slot_states[index]
    if state != SLOT_FREE &&
      state != SLOT_RETIRED &&
      state != SLOT_CANCELLED &&
      state != SLOT_DEADLINE &&
      state != SLOT_FAILED &&
      now_millis >= self.deadlines[index] {
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
          request_id,
          self.request_generations[index],
          DeadlineReached,
        )
      }
      expired += 1
    }
  }
  expired
}
