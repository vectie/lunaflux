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
pub fn Scheduler::model_identity(self : Scheduler) -> @spec.ModelIdentity {
  self.model_identity
}

///|
pub fn Scheduler::last_submitted_plan_sequence(self : Scheduler) -> UInt64 {
  self.last_plan_sequence
}

///|
pub fn Scheduler::last_retired_plan_sequence(self : Scheduler) -> UInt64 {
  self.last_retired_plan_sequence
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
pub fn Scheduler::generated_token_count(self : Scheduler) -> Int {
  self.output_count
}

///|
/// Allocation-sensitive terminal dequeue. Use `terminal_notice_count` to
/// avoid the typed empty-ring error on the production path.
pub fn Scheduler::take_terminal_notice(
  self : Scheduler,
) -> TerminalNotice raise SchedulerError {
  if self.terminal_count == 0 {
    raise invalid(TerminalNoticeCapacity, NoRunnableWork)
  }
  let index = self.terminal_head
  let notice = TerminalNotice(
    self.terminal_request_ids[index],
    self.terminal_generations[index],
    self.terminal_reasons[index],
  )
  self.terminal_head = (index + 1) % self.terminal_request_ids.length()
  self.terminal_count -= 1
  notice
}

///|
pub fn Scheduler::take_generated_token(
  self : Scheduler,
) -> GeneratedTokenNotice raise SchedulerError {
  if self.output_count == 0 {
    raise invalid(OutputCapacity, NoRunnableWork)
  }
  let index = self.output_head
  let notice = GeneratedTokenNotice(
    self.output_request_ids[index],
    self.output_generations[index],
    self.output_token_ids[index],
    self.output_positions[index],
  )
  self.output_head = (index + 1) % self.output_request_ids.length()
  self.output_count -= 1
  notice
}

///|
fn Scheduler::preflight_active_release(
  self : Scheduler,
  slot : Int,
) -> Unit raise SchedulerError {
  if slot < 0 ||
    slot >= self.slot_states.length() ||
    self.slot_states[slot] == SLOT_WAITING ||
    self.slot_states[slot] == SLOT_FREE ||
    self.slot_states[slot] == SLOT_RETIRED {
    raise invalid(RequestState, InvalidTransition)
  }
  let mut found = false
  for index in 0..<self.active_count {
    if self.active_slots[index] == slot {
      if found {
        raise invalid(RequestState, InvalidTransition)
      }
      found = true
    }
  }
  if !found {
    raise invalid(RequestState, InvalidTransition)
  }
  let table = try self.request_tables.read(slot) catch {
    error => raise BlockTable(error)
  } noraise {
    value => value
  }
  let page_count = try self.block_tables.page_count(table) catch {
    error => raise BlockTable(error)
  } noraise {
    value => value
  }
  if page_count > self.release_pages.capacity() {
    raise invalid(PageCapacity, OutOfRange)
  }
  for index in 0..<page_count {
    let page = try self.block_tables.page_at(table, index) catch {
      error => raise BlockTable(error)
    } noraise {
      value => value
    }
    let references = try
      self.page_allocator.active_reference_count(page)
    catch {
      error => raise PageAllocator(error)
    } noraise {
      value => value
    }
    if references <= 0 {
      raise invalid(PageCapacity, InvalidTransition)
    }
  }
}

///|
fn Scheduler::remove_active(self : Scheduler, slot : Int) -> Unit {
  let mut index = 0
  while index < self.active_count && self.active_slots[index] != slot {
    index += 1
  }
  if index == self.active_count {
    abort("scheduler active ownership invariant")
  }
  for next in (index + 1)..<self.active_count {
    self.active_slots[next - 1] = self.active_slots[next]
  }
  self.active_count -= 1
  self.active_slots[self.active_count] = NO_INDEX
}

///|
fn Scheduler::release_active_request(self : Scheduler, slot : Int) -> Unit {
  let table = try! self.request_tables.read(slot)
  try! self.block_tables.reset_and_release(table, self.release_pages)
  for index in 0..<self.release_pages.length() {
    try! self.page_allocator.release_active(
      try! self.release_pages.page_at(index),
    )
  }
  try! self.request_tables.clear(slot)
  self.remove_active(slot)
  self.recycle_slot(slot)
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
  if !self.in_flight[index] {
    self.preflight_active_release(index)
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
  if !self.in_flight[index] {
    self.release_active_request(index)
  }
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
  for index in 0..<self.slot_states.length() {
    let state = self.slot_states[index]
    if state != SLOT_FREE &&
      state != SLOT_RETIRED &&
      state != SLOT_WAITING &&
      state != SLOT_CANCELLED &&
      state != SLOT_DEADLINE &&
      state != SLOT_FAILED &&
      !self.in_flight[index] &&
      now_millis >= self.deadlines[index] {
      self.preflight_active_release(index)
    }
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
        if !self.in_flight[index] {
          self.release_active_request(index)
        }
      }
      expired += 1
    }
  }
  expired
}
