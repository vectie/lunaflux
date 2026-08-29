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
/// Returns the opaque caller route attached to the globally oldest
/// publication without consuming either publication ring.
pub fn Scheduler::next_publication_route(
  self : Scheduler,
) -> SchedulerPublicationRoute raise SchedulerError {
  let (lane, generation) = match self.next_publication_kind() {
    GeneratedTokenPublication =>
      (
        self.output_route_lanes[self.output_head],
        self.output_route_generations[self.output_head],
      )
    TerminalPublication =>
      (
        self.terminal_route_lanes[self.terminal_head],
        self.terminal_route_generations[self.terminal_head],
      )
  }
  SchedulerPublicationRoute::new(lane, generation)
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
      let slot_index = self.output_slot_indices[self.output_head]
      let slot_generation = self.output_slot_generations[self.output_head]
      let route = (self.output_route_generations[self.output_head] << 32) |
        self.output_route_lanes[self.output_head].to_uint64()
      let notice = self.take_generated_token()
      SchedulerPublication(
        scheduler_publication_owner(0, slot_index, slot_generation),
        notice.request_id(),
        notice.request_generation(),
        (notice.position().to_uint64() << 32) | notice.token_id().to_uint64(),
        notice.publication_sequence(),
        route,
      )
    }
    TerminalPublication => {
      let slot_index = self.terminal_slot_indices[self.terminal_head]
      let slot_generation = self.terminal_slot_generations[self.terminal_head]
      let route = (self.terminal_route_generations[self.terminal_head] << 32) |
        self.terminal_route_lanes[self.terminal_head].to_uint64()
      let notice = self.take_terminal_notice()
      SchedulerPublication(
        scheduler_publication_owner(1, slot_index, slot_generation),
        notice.request_id(),
        notice.request_generation(),
        (notice.cached_input_tokens().to_uint64() << 32) |
        terminal_reason_code(notice.reason()).to_uint64(),
        notice.publication_sequence(),
        route,
      )
    }
  }
}

///|
/// Invalidates every active request after loss of the worker-owned device KV
/// generation. Waiting requests remain logically queued, but any shared prefix
/// pages/table are released and their replay cursor returns to token zero.
/// Every cached prefix anchor is evicted before a fresh worker may run. The
/// operation is retryable under terminal-publication backpressure and mutates
/// nothing until every active request and KV release is proven.
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
  self.preflight_waiting_prefix_device_loss()
  for _ in 0..<count {
    let slot = self.active_slots[0]
    let request = match self.requests[slot] {
      Some(value) => value
      None => abort("scheduler device-state invalidation invariant")
    }
    self.request_generations[slot] = self.allocate_request_generation()
    self.slot_states[slot] = SLOT_FAILED
    self.enqueue_terminal(
      slot,
      request.request_id(),
      self.request_generations[slot],
      WorkerFailed,
    )
    self.release_active_request(slot)
  }
  self.release_waiting_prefix_device_state()
  self.clear_cached_prefix_device_state()
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
    self.terminal_cached_input_tokens[index],
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
  let position = self.active_positions[slot]
  if position < 0 ||
    position >= self.active_count ||
    self.active_slots[position] != slot {
    raise invalid(RequestState, InvalidTransition)
  }
  self.preflight_table_page_release(slot)
  self.preflight_prefix_entry_release(slot)
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
  self.release_prefix_entry(slot)
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
  self.release_waiting_prefix(index)
  self.remove_waiting(index)
  self.slot_states[index] = state
  self.request_generations[index] = self.allocate_request_generation()
  self.enqueue_terminal(
    index,
    request_id,
    self.request_generations[index],
    reason,
  )
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
    index,
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
