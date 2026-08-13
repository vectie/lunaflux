///|
fn Scheduler::require_no_cancel_reservation(
  self : Scheduler,
) -> Unit raise SchedulerError {
  if self.cancel_reserved {
    raise invalid(CancellationReservation, InvalidTransition)
  }
}

///|
fn Scheduler::has_exact_terminal(
  self : Scheduler,
  handle : RequestHandle,
) -> Bool {
  for offset in 0..<self.terminal_count {
    let index = (self.terminal_head + offset) %
      self.terminal_request_ids.length()
    if self.terminal_request_ids[index] == handle.request_id() &&
      self.terminal_generations[index] == handle.request_generation() {
      return true
    }
  }
  false
}

///|
fn Scheduler::clear_cancel_reservation(self : Scheduler) -> Unit {
  self.cancel_reserved = false
  self.cancel_slot = NO_INDEX
  self.cancel_slot_generation = 0
  self.cancel_request_generation = @inference.RequestGeneration::initial()
  self.cancel_publication_cursor = 0UL
  self.cancel_step = 0UL
  self.cancel_token_taken = false
}

///|
/// Reserves every fallible condition needed to cancel one exact live request.
/// While active, every other scheduler mutation is rejected; the sole allowed
/// mutation is dequeueing the globally next generated notice for this exact
/// request through `take_reserved_generated_token`.
pub fn Scheduler::reserve_cancel_exact(
  self : Scheduler,
  handle : RequestHandle,
) -> CancelReservationStart raise SchedulerError {
  self.require_no_cancel_reservation()
  if self.has_exact_terminal(handle) {
    return CancelNaturalTerminal
  }
  let index = self.checked_handle_index(handle)
  if self.slot_states[index] == SLOT_CANCELLED ||
    self.slot_states[index] == SLOT_DEADLINE ||
    self.slot_states[index] == SLOT_FAILED ||
    self.slot_states[index] == SLOT_FINISHING {
    raise invalid(RequestState, InvalidTransition)
  }
  if self.terminal_available() == 0 {
    raise invalid(TerminalNoticeCapacity, Backpressured)
  }
  self.preflight_publications(1)
  if self.generation_cursor == MAX_GENERATION {
    raise invalid(RequestGeneration, Exhausted)
  }
  if self.slot_states[index] != SLOT_WAITING && !self.in_flight[index] {
    self.preflight_active_release(index)
  }
  self.cancel_slot = index
  self.cancel_slot_generation = handle.slot().generation()
  self.cancel_request_id = handle.request_id()
  self.cancel_request_generation = handle.request_generation()
  self.cancel_publication_cursor = self.publication_cursor
  self.cancel_step = self.step
  self.cancel_token_taken = false
  self.cancel_reserved = true
  CancelReserved
}

///|
/// Dequeues only the globally oldest generated notice and only when it is
/// bound to the reserved request generation. The reservation remains active.
pub fn Scheduler::take_reserved_generated_token(
  self : Scheduler,
) -> GeneratedTokenNotice raise SchedulerError {
  if !self.cancel_reserved ||
    self.cancel_token_taken ||
    self.publication_cursor != self.cancel_publication_cursor ||
    self.step != self.cancel_step ||
    self.next_publication_kind() != GeneratedTokenPublication {
    raise invalid(CancellationReservation, InvalidTransition)
  }
  let index = self.output_head
  if self.output_request_ids[index] != self.cancel_request_id ||
    self.output_generations[index] != self.cancel_request_generation {
    raise invalid(CancellationReservation, Mismatch)
  }
  let notice = GeneratedTokenNotice(
    self.output_request_ids[index],
    self.output_generations[index],
    self.output_token_ids[index],
    self.output_positions[index],
    self.output_sequences[index],
  )
  self.output_head = (index + 1) % self.output_request_ids.length()
  self.output_count -= 1
  self.cancel_token_taken = true
  notice
}

///|
/// Commits a successfully preflighted cancellation without a fallible branch.
/// The scheduler reason is always physical `CancelledByCaller`; a higher
/// session owner may bind a private cut cause and translate only the resulting
/// exact-generation terminal notice into string-stop or output-failure policy.
/// `CancelReservationAbsent` makes repeated or aliased commits harmless.
pub fn Scheduler::commit_cancel_reservation(
  self : Scheduler,
) -> CancelReservationCommit {
  if !self.cancel_reserved {
    return CancelReservationAbsent
  }
  if self.publication_cursor != self.cancel_publication_cursor ||
    self.step != self.cancel_step ||
    self.cancel_slot < 0 ||
    self.cancel_slot >= self.slot_states.length() ||
    self.slot_generations[self.cancel_slot] != self.cancel_slot_generation ||
    self.request_generations[self.cancel_slot] != self.cancel_request_generation {
    abort("scheduler cancellation reservation invariant")
  }
  let index = self.cancel_slot
  let generation = try! self.allocate_request_generation()
  if self.slot_states[index] == SLOT_WAITING {
    self.remove_waiting(index)
    self.slot_states[index] = SLOT_CANCELLED
    self.request_generations[index] = generation
    try! self.enqueue_terminal(
      self.cancel_request_id,
      generation,
      CancelledByCaller,
    )
    self.recycle_slot(index)
  } else {
    self.request_generations[index] = generation
    self.slot_states[index] = SLOT_CANCELLED
    try! self.enqueue_terminal(
      self.cancel_request_id,
      generation,
      CancelledByCaller,
    )
    if !self.in_flight[index] {
      self.release_active_request(index)
    }
  }
  self.clear_cancel_reservation()
  CancelCommitted(generation)
}

///|
/// Releases a reservation without changing request, publication, KV, or plan
/// state. Repeated abort is an exact no-op.
pub fn Scheduler::abort_cancel_reservation(self : Scheduler) -> Unit {
  if self.cancel_reserved {
    self.clear_cancel_reservation()
  }
}
