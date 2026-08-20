///|
fn Scheduler::preflight_admission_request(
  self : Scheduler,
  request : TokenizedRequest,
) -> Unit raise SchedulerError {
  self.require_no_cancel_reservation()
  if request.model_identity() != self.model_identity {
    raise invalid(ModelIdentity, Mismatch)
  }
  if self.find_request(request.request_id()) != NO_INDEX {
    raise invalid(Request, Duplicate)
  }
  if self.waiting_count >= self.capacity.max_waiting_requests() {
    raise invalid(WaitingCapacity, Exhausted)
  }
  if self.free_head == NO_INDEX {
    raise invalid(RequestCapacity, Exhausted)
  }
  if self.generation_cursor == MAX_GENERATION {
    raise invalid(RequestGeneration, Exhausted)
  }
  let context = request.effective_limits().context_ceiling()
  let prompt_tokens = request.input_tokens().length()
  let maximum_token = request.input_maximum_token_status(
    self.capacity.worker_max_token_id(),
  )
  if maximum_token.is_stale() {
    raise invalid(Request, Stale)
  }
  if maximum_token.is_out_of_range() || !maximum_token.is_within_bound() {
    raise invalid(Request, OutOfRange)
  }
  let maximum_sequence_tokens = prompt_tokens +
    request.effective_limits().max_new_tokens()
  let prompt_pages = ceiling_pages(
    prompt_tokens,
    self.capacity.tokens_per_page(),
  )
  let maximum_sequence_pages = ceiling_pages(
    maximum_sequence_tokens,
    self.capacity.tokens_per_page(),
  )
  if context > self.capacity.model_max_sequence_tokens() ||
    ceiling_pages(context, self.capacity.tokens_per_page()) >
    self.capacity.block_table_pages_per_request() ||
    prompt_pages >
    self.capacity.total_page_count() -
    self.capacity.emergency_decode_page_reserve() ||
    maximum_sequence_pages > self.capacity.total_page_count() {
    raise invalid(Context, OutOfRange)
  }
}

///|
fn Scheduler::install_admission_request(
  self : Scheduler,
  request : TokenizedRequest,
) -> Int {
  let deadline = request.admission_deadline().at_millis()
  let generation = try! self.allocate_request_generation()
  let index = try! self.allocate_slot()
  self.slot_states[index] = SLOT_WAITING
  self.requests[index] = Some(request)
  self.request_generations[index] = generation
  self.deadlines[index] = deadline
  self.admitted_steps[index] = self.step
  self.live_count += 1
  self.enqueue_waiting(index)
  index
}

///|
fn Scheduler::commit_admission_request(
  self : Scheduler,
  request : TokenizedRequest,
  now_millis : UInt64,
) -> Unit raise SchedulerError {
  self.preflight_admission_request(request)
  if request.admission_deadline().is_expired(now_millis) {
    raise invalid(Deadline, Expired)
  }
  ignore(self.install_admission_request(request))
}

///|
pub fn Scheduler::prepare_admission(
  self : Scheduler,
  request : TokenizedRequest,
) -> PreparedAdmission {
  {
    owner: self,
    request,
    handle: {
      slot: SchedulerSlotId(NO_INDEX, 0),
      request_id: request.request_id(),
      request_generation: @inference.RequestGeneration::initial(),
    },
    committed: false,
  }
}

///|
pub fn Scheduler::commit_admission(
  self : Scheduler,
  prepared : PreparedAdmission,
  now_millis : UInt64,
) -> Unit raise SchedulerError {
  if !physical_equal(prepared.owner, self) || prepared.committed {
    raise invalid(Request, InvalidTransition)
  }
  self.commit_admission_request(prepared.request, now_millis)
  let index = self.find_request(prepared.request.request_id())
  if index == NO_INDEX {
    abort("prepared scheduler admission invariant")
  }
  prepared.handle.slot = SchedulerSlotId(index, self.slot_generations[index])
  prepared.handle.request_generation = self.request_generations[index]
  prepared.committed = true
}

///|
pub fn PreparedAdmission::request_handle(
  self : PreparedAdmission,
) -> RequestHandle raise SchedulerError {
  if !self.committed {
    raise invalid(Request, InvalidTransition)
  }
  self.handle
}

///|
pub fn Scheduler::prepare_exclusive_admission(
  self : Scheduler,
  request : TokenizedRequest,
) -> PreparedExclusiveAdmission raise SchedulerError {
  self.preflight_admission_request(request)
  let prepared : PreparedExclusiveAdmission = {
    owner: self,
    request,
    handle: {
      slot: SchedulerSlotId(NO_INDEX, 0),
      request_id: request.request_id(),
      request_generation: @inference.RequestGeneration::initial(),
    },
    committed: false,
    consumed: false,
  }
  self.exclusive_admission_reserved = true
  prepared
}

///|
/// Commits the exclusively retained shell after reauthenticating its token
/// backing. No scheduler mutation can intervene, but an outer retained Luna
/// lease alias may invalidate the backing while rooted startup is in progress.
pub fn Scheduler::commit_exclusive_admission(
  self : Scheduler,
  prepared : PreparedExclusiveAdmission,
  now_millis : UInt64,
) -> ExclusiveAdmissionCommit {
  if !physical_equal(prepared.owner, self) ||
    prepared.consumed ||
    !self.exclusive_admission_reserved {
    return ExclusiveAdmissionInvalidated
  }
  let maximum_token = prepared.request.input_maximum_token_status(
    self.capacity.worker_max_token_id(),
  )
  if !maximum_token.is_within_bound() {
    self.exclusive_admission_reserved = false
    prepared.consumed = true
    return ExclusiveAdmissionInvalidated
  }
  if prepared.request.admission_deadline().is_expired(now_millis) {
    self.exclusive_admission_reserved = false
    prepared.consumed = true
    return ExclusiveAdmissionExpired
  }
  let index = self.install_admission_request(prepared.request)
  prepared.handle.slot = SchedulerSlotId(index, self.slot_generations[index])
  prepared.handle.request_generation = self.request_generations[index]
  prepared.committed = true
  prepared.consumed = true
  self.exclusive_admission_reserved = false
  ExclusiveAdmissionCommitted
}

///|
pub fn Scheduler::abort_exclusive_admission(
  self : Scheduler,
  prepared : PreparedExclusiveAdmission,
) -> Unit {
  if !physical_equal(prepared.owner, self) ||
    prepared.consumed ||
    !self.exclusive_admission_reserved {
    return
  }
  self.exclusive_admission_reserved = false
  prepared.consumed = true
}

///|
pub fn Scheduler::has_exclusive_admission(
  self : Scheduler,
  prepared : PreparedExclusiveAdmission,
) -> Bool {
  physical_equal(prepared.owner, self) &&
  self.exclusive_admission_reserved &&
  !prepared.consumed
}

///|
pub fn PreparedExclusiveAdmission::request_handle(
  self : PreparedExclusiveAdmission,
) -> RequestHandle {
  if !self.committed {
    abort("exclusive scheduler handle invariant")
  }
  self.handle
}

///|
pub fn Scheduler::admit(
  self : Scheduler,
  request : TokenizedRequest,
  now_millis : UInt64,
) -> RequestHandle raise SchedulerError {
  let prepared = self.prepare_admission(request)
  self.commit_admission(prepared, now_millis)
  prepared.request_handle()
}
