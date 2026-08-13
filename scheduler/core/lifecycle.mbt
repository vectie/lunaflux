///|
/// Returns the fixed ring that owns the globally oldest publication without
/// consuming it. Following this head preserves the exact token/terminal commit
/// order across the otherwise independent bounded rings.
pub fn Scheduler::next_publication_kind(
  self : Scheduler,
) -> SchedulerPublicationKind raise SchedulerError {
  if self.output_count == 0 && self.terminal_count == 0 {
    raise invalid(OutputCapacity, NoRunnableWork)
  }
  if self.output_count == 0 {
    return TerminalPublication
  }
  if self.terminal_count == 0 {
    return GeneratedTokenPublication
  }
  if self.output_sequences[self.output_head] <
    self.terminal_sequences[self.terminal_head] {
    GeneratedTokenPublication
  } else {
    TerminalPublication
  }
}

///|
/// Consumes the globally oldest scheduler publication in one operation. This
/// prevents a caller from observing the correct ring and then accidentally
/// dequeuing the other one.
pub fn Scheduler::take_next_publication(
  self : Scheduler,
) -> SchedulerPublication raise SchedulerError {
  self.require_no_cancel_reservation()
  match self.next_publication_kind() {
    GeneratedTokenPublication => {
      let notice = self.take_generated_token()
      SchedulerPublication(
        0,
        notice.request_id(),
        notice.request_generation(),
        notice.token_id(),
        notice.position(),
        notice.publication_sequence(),
      )
    }
    TerminalPublication => {
      let notice = self.take_terminal_notice()
      SchedulerPublication(
        1,
        notice.request_id(),
        notice.request_generation(),
        0,
        terminal_reason_code(notice.reason()),
        notice.publication_sequence(),
      )
    }
  }
}

///|
/// Invalidates every active request after loss of the worker-owned device KV
/// generation. Waiting requests remain queued because they own no device
/// state. The operation is retryable under terminal-publication backpressure
/// and mutates nothing until every active request and KV release is proven.
/// No submitted plan may remain: callers must first retire each exact process
/// obligation through `complete` or `fail_plan`.
pub fn Scheduler::invalidate_device_state(
  self : Scheduler,
) -> Int raise SchedulerError {
  self.require_no_cancel_reservation()
  if self.plan_a_sequence != 0UL ||
    self.plan_b_sequence != 0UL ||
    self.last_retired_plan_sequence != self.last_plan_sequence {
    raise invalid(Plan, InvalidTransition)
  }
  let count = self.active_count
  if count == 0 {
    return 0
  }
  if count > self.terminal_available() {
    raise invalid(TerminalNoticeCapacity, Backpressured)
  }
  self.preflight_publications(count)
  if self.generation_cursor > MAX_GENERATION - count.to_uint64() {
    raise invalid(RequestGeneration, Exhausted)
  }
  for index in 0..<count {
    let slot = self.active_slots[index]
    if slot < 0 || slot >= self.slot_states.length() || self.in_flight[slot] {
      raise invalid(RequestState, InvalidTransition)
    }
    self.preflight_active_release(slot)
  }
  for _ in 0..<count {
    let slot = self.active_slots[0]
    let request = match self.requests[slot] {
      Some(value) => value
      None => abort("scheduler device-state invalidation invariant")
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
  count
}

///|
/// Invalidates device state while returning the exact fresh failure generation
/// for one authenticated request when that request was active.
pub fn Scheduler::invalidate_device_state_exact(
  self : Scheduler,
  handle : RequestHandle,
) -> ExactDeviceInvalidation raise SchedulerError {
  for offset in 0..<self.terminal_count {
    let notice_index = (self.terminal_head + offset) %
      self.terminal_request_ids.length()
    if self.terminal_request_ids[notice_index] == handle.request_id() &&
      self.terminal_generations[notice_index] == handle.request_generation() {
      if self.live_count != 0 {
        raise invalid(RequestCapacity, Mismatch)
      }
      ignore(self.invalidate_device_state())
      return ExactDeviceAlreadyTerminal(self.terminal_generations[notice_index])
    }
  }
  if self.live_count != 1 {
    raise invalid(RequestCapacity, Mismatch)
  }
  let index = self.checked_handle_index(handle)
  let was_active = self.slot_states[index] != SLOT_WAITING
  let count = self.invalidate_device_state()
  if !was_active {
    return ExactDeviceNoChange
  }
  let expected_generation = @inference.RequestGeneration::from_wire(
    self.generation_cursor,
  )
  for offset in 0..<self.terminal_count {
    let notice_index = (self.terminal_head + offset) %
      self.terminal_request_ids.length()
    if self.terminal_request_ids[notice_index] == handle.request_id() &&
      self.terminal_generations[notice_index] == expected_generation &&
      self.terminal_reasons[notice_index] == WorkerFailed {
      return ExactDeviceInvalidated(
        count,
        self.terminal_generations[notice_index],
      )
    }
  }
  abort("exact device invalidation terminal invariant")
}

///|
/// Allocation-sensitive terminal dequeue. Use `terminal_notice_count` to
/// avoid the typed empty-ring error on the production path.
pub fn Scheduler::take_terminal_notice(
  self : Scheduler,
) -> TerminalNotice raise SchedulerError {
  self.require_no_cancel_reservation()
  if self.terminal_count == 0 {
    raise invalid(TerminalNoticeCapacity, NoRunnableWork)
  }
  if self.output_count > 0 &&
    self.output_sequences[self.output_head] <
    self.terminal_sequences[self.terminal_head] {
    raise invalid(PublicationSequence, OutOfOrder)
  }
  let index = self.terminal_head
  let notice = TerminalNotice(
    self.terminal_request_ids[index],
    self.terminal_generations[index],
    self.terminal_reasons[index],
    self.terminal_sequences[index],
  )
  self.terminal_head = (index + 1) % self.terminal_request_ids.length()
  self.terminal_count -= 1
  notice
}

///|
pub fn Scheduler::take_generated_token(
  self : Scheduler,
) -> GeneratedTokenNotice raise SchedulerError {
  self.require_no_cancel_reservation()
  if self.output_count == 0 {
    raise invalid(OutputCapacity, NoRunnableWork)
  }
  if self.terminal_count > 0 &&
    self.terminal_sequences[self.terminal_head] <
    self.output_sequences[self.output_head] {
    raise invalid(PublicationSequence, OutOfOrder)
  }
  let index = self.output_head
  let notice = GeneratedTokenNotice(
    self.output_request_ids[index],
    self.output_generations[index],
    self.output_token_ids[index],
    self.output_positions[index],
    self.output_sequences[index],
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
pub fn Scheduler::cancel_exact(
  self : Scheduler,
  handle : RequestHandle,
) -> @inference.RequestGeneration raise SchedulerError {
  self.require_no_cancel_reservation()
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
  self.preflight_publications(1)
  if self.slot_states[index] == SLOT_WAITING {
    self.terminate_waiting(index, SLOT_CANCELLED, CancelledByCaller)
    return self.request_generations[index]
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
  self.request_generations[index]
}

///|
pub fn Scheduler::cancel(
  self : Scheduler,
  handle : RequestHandle,
) -> Unit raise SchedulerError {
  self.require_no_cancel_reservation()
  ignore(self.cancel_exact(handle))
}

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
  self.preflight_publications(candidate_count)
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
