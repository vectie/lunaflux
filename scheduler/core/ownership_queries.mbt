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
pub fn Scheduler::worker_protocol_limits(
  self : Scheduler,
) -> @worker_protocol.WorkerProtocolLimits {
  self.worker_limits
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
pub fn Scheduler::is_pristine(self : Scheduler) -> Bool {
  self.live_count == 0 &&
  self.waiting_count == 0 &&
  self.active_count == 0 &&
  self.output_count == 0 &&
  self.terminal_count == 0 &&
  self.publication_cursor == 0UL &&
  self.generation_cursor == 0UL &&
  self.step == 0UL &&
  self.last_plan_sequence == 0UL &&
  self.last_retired_plan_sequence == 0UL &&
  self.plan_a_sequence == 0UL &&
  self.plan_b_sequence == 0UL &&
  !self.cancel_reserved &&
  !self.exclusive_admission_reserved
}

///|
pub fn Scheduler::is_quiescent(self : Scheduler) -> Bool {
  self.live_count == 0 &&
  self.waiting_count == 0 &&
  self.active_count == 0 &&
  self.output_count == 0 &&
  self.terminal_count == 0 &&
  self.last_plan_sequence == self.last_retired_plan_sequence &&
  self.plan_a_sequence == 0UL &&
  self.plan_b_sequence == 0UL &&
  !self.cancel_reserved &&
  !self.exclusive_admission_reserved
}
