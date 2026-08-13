///|
fn Scheduler::commit_admission_request(
  self : Scheduler,
  request : TokenizedRequest,
  now_millis : UInt64,
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
  let context = request.effective_limits().context_ceiling()
  let prompt_tokens = request.input_tokens().length()
  for token in request.input_tokens().token_ids() {
    if token > self.capacity.worker_max_token_id() {
      raise invalid(Request, OutOfRange)
    }
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
  let deadline = request.admission_deadline().at_millis()
  if request.admission_deadline().is_expired(now_millis) {
    raise invalid(Deadline, Expired)
  }
  let generation = self.allocate_request_generation()
  let index = self.allocate_slot()
  self.slot_states[index] = SLOT_WAITING
  self.requests[index] = Some(request)
  self.request_generations[index] = generation
  self.deadlines[index] = deadline
  self.admitted_steps[index] = self.step
  self.live_count += 1
  self.enqueue_waiting(index)
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
pub fn Scheduler::admit(
  self : Scheduler,
  request : TokenizedRequest,
  now_millis : UInt64,
) -> RequestHandle raise SchedulerError {
  let prepared = self.prepare_admission(request)
  self.commit_admission(prepared, now_millis)
  prepared.request_handle()
}
