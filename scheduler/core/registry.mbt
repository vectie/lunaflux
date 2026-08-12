///|
fn Scheduler::allocate_request_generation(
  self : Scheduler,
) -> @inference.RequestGeneration raise SchedulerError {
  if self.generation_cursor == MAX_GENERATION {
    raise invalid(RequestGeneration, Exhausted)
  }
  self.generation_cursor += 1UL
  @inference.RequestGeneration::from_wire(self.generation_cursor)
}

///|
fn Scheduler::find_request(self : Scheduler, id : @inference.RequestId) -> Int {
  for index in 0..<self.slot_states.length() {
    if self.slot_states[index] != SLOT_FREE &&
      self.slot_states[index] != SLOT_RETIRED {
      match self.requests[index] {
        Some(request) if request.request_id() == id => return index
        _ => ()
      }
    }
  }
  NO_INDEX
}

///|
fn Scheduler::allocate_slot(self : Scheduler) -> Int raise SchedulerError {
  if self.free_head == NO_INDEX {
    raise invalid(RequestCapacity, Exhausted)
  }
  let index = self.free_head
  self.free_head = self.free_next[index]
  if self.free_head == NO_INDEX {
    self.free_tail = NO_INDEX
  }
  self.free_next[index] = NO_INDEX
  self.slot_generations[index] += 1
  index
}

///|
fn Scheduler::enqueue_waiting(self : Scheduler, index : Int) -> Unit {
  self.waiting_next[index] = NO_INDEX
  if self.waiting_tail == NO_INDEX {
    self.waiting_head = index
    self.waiting_tail = index
  } else {
    self.waiting_next[self.waiting_tail] = index
    self.waiting_tail = index
  }
  self.waiting_count += 1
}

///|
fn Scheduler::remove_waiting(self : Scheduler, index : Int) -> Unit {
  let mut previous = NO_INDEX
  let mut current = self.waiting_head
  while current != NO_INDEX {
    if current == index {
      let next = self.waiting_next[current]
      if previous == NO_INDEX {
        self.waiting_head = next
      } else {
        self.waiting_next[previous] = next
      }
      if self.waiting_tail == current {
        self.waiting_tail = previous
      }
      self.waiting_next[current] = NO_INDEX
      self.waiting_count -= 1
      return
    }
    previous = current
    current = self.waiting_next[current]
  }
}

///|
fn Scheduler::recycle_slot(self : Scheduler, index : Int) -> Unit {
  self.requests[index] = None
  self.deadlines[index] = 0UL
  self.admitted_steps[index] = 0UL
  self.processed_inputs[index] = 0
  self.output_counts[index] = 0
  self.in_flight[index] = false
  self.waiting_next[index] = NO_INDEX
  self.live_count -= 1
  if self.slot_generations[index] == MAX_SLOT_GENERATION {
    self.slot_states[index] = SLOT_RETIRED
    self.retired_slot_count += 1
    return
  }
  self.slot_states[index] = SLOT_FREE
  self.free_next[index] = NO_INDEX
  if self.free_tail == NO_INDEX {
    self.free_head = index
    self.free_tail = index
  } else {
    self.free_next[self.free_tail] = index
    self.free_tail = index
  }
}

///|
fn Scheduler::terminal_available(self : Scheduler) -> Int {
  self.terminal_request_ids.length() - self.terminal_count
}

///|
fn Scheduler::enqueue_terminal(
  self : Scheduler,
  request_id : @inference.RequestId,
  generation : @inference.RequestGeneration,
  reason : TerminalReason,
) -> Unit raise SchedulerError {
  if self.terminal_available() == 0 {
    raise invalid(TerminalNoticeCapacity, Backpressured)
  }
  let index = self.terminal_tail
  self.terminal_request_ids[index] = request_id
  self.terminal_generations[index] = generation
  self.terminal_reasons[index] = reason
  self.terminal_tail = (index + 1) % self.terminal_request_ids.length()
  self.terminal_count += 1
}

///|
fn ceiling_pages(tokens : Int, tokens_per_page : Int) -> Int {
  (tokens - 1) / tokens_per_page + 1
}

///|
pub fn Scheduler::new(
  capacity : @runtime_resolved.ResolvedRuntimeCapacity,
  model_identity : @spec.ModelIdentity,
  model_plan_generation : @worker_protocol.ModelPlanGeneration,
  prefill_recipe : @worker_protocol.RowCapabilityRecipe,
  decode_recipe : @worker_protocol.RowCapabilityRecipe,
) -> Scheduler raise SchedulerError {
  if prefill_recipe.identity() != model_identity ||
    decode_recipe.identity() != model_identity ||
    prefill_recipe.model_plan_generation() != model_plan_generation ||
    decode_recipe.model_plan_generation() != model_plan_generation {
    raise invalid(ModelIdentity, Mismatch)
  }
  let rows = capacity.max_rows_per_step()
  let max_recipe_length = if prefill_recipe.length() > decode_recipe.length() {
    prefill_recipe.length()
  } else {
    decode_recipe.length()
  }
  if rows > capacity.worker_max_capabilities() / max_recipe_length {
    raise invalid(Plan, OutOfRange)
  }
  let slots = capacity.total_request_slots()
  let placeholder_id = try @inference.RequestId::new(1UL) catch {
    _ => raise invalid(Request, OutOfRange)
  } noraise {
    value => value
  }
  let free_next = FixedArray::make(slots, NO_INDEX)
  for index in 0..<slots {
    free_next[index] = if index + 1 < slots { index + 1 } else { NO_INDEX }
  }
  let page_allocator = @page_allocator.PageAllocator::new(
    capacity.page_allocator_limits(),
  )
  let block_tables = try
    @block_table.BlockTableArena::new(capacity.block_table_limits())
  catch {
    error => raise BlockTable(error)
  } noraise {
    arena => arena
  }
  {
    capacity,
    model_identity,
    page_allocator,
    block_tables,
    model_plan_generation,
    slot_states: FixedArray::make(slots, SLOT_FREE),
    slot_generations: FixedArray::make(slots, 0),
    requests: FixedArray::make(slots, None),
    request_generations: FixedArray::make(
      slots,
      @inference.RequestGeneration::initial(),
    ),
    deadlines: FixedArray::make(slots, 0UL),
    admitted_steps: FixedArray::make(slots, 0UL),
    processed_inputs: FixedArray::make(slots, 0),
    output_counts: FixedArray::make(slots, 0),
    in_flight: FixedArray::make(slots, false),
    waiting_next: FixedArray::make(slots, NO_INDEX),
    free_next,
    terminal_request_ids: FixedArray::make(slots, placeholder_id),
    terminal_generations: FixedArray::make(
      slots,
      @inference.RequestGeneration::initial(),
    ),
    terminal_reasons: FixedArray::make(slots, CancelledByCaller),
    free_head: 0,
    free_tail: slots - 1,
    waiting_head: NO_INDEX,
    waiting_tail: NO_INDEX,
    waiting_count: 0,
    live_count: 0,
    retired_slot_count: 0,
    terminal_head: 0,
    terminal_tail: 0,
    terminal_count: 0,
    generation_cursor: 0UL,
    step: 0UL,
  }
}

///|
pub fn Scheduler::admit(
  self : Scheduler,
  request : TokenizedRequest,
  now_millis : UInt64,
) -> RequestHandle raise SchedulerError {
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
  let budget = request.deadline_budget().milliseconds()
  if now_millis > MAX_GENERATION - budget {
    raise invalid(Deadline, Overflow)
  }
  let deadline = now_millis + budget
  if deadline <= now_millis {
    raise invalid(Deadline, Expired)
  }
  let generation = self.allocate_request_generation()
  let index = self.allocate_slot()
  let request_id = request.request_id()
  self.slot_states[index] = SLOT_WAITING
  self.requests[index] = Some(request)
  self.request_generations[index] = generation
  self.deadlines[index] = deadline
  self.admitted_steps[index] = self.step
  self.live_count += 1
  self.enqueue_waiting(index)
  {
    slot: SchedulerSlotId(index, self.slot_generations[index]),
    request_id,
    request_generation: generation,
  }
}

///|
fn Scheduler::checked_handle_index(
  self : Scheduler,
  handle : RequestHandle,
) -> Int raise SchedulerError {
  let index = handle.slot().index()
  if index < 0 ||
    index >= self.slot_states.length() ||
    self.slot_generations[index] != handle.slot().generation() ||
    self.slot_states[index] == SLOT_FREE ||
    self.slot_states[index] == SLOT_RETIRED {
    raise invalid(Request, Stale)
  }
  guard self.requests[index] is Some(request) else {
    raise invalid(RequestState, InvalidTransition)
  }
  if request.request_id() != handle.request_id() ||
    self.request_generations[index] != handle.request_generation() {
    raise invalid(Request, Stale)
  }
  index
}

///|
fn scheduler_state(state : Int) -> SchedulerRequestState {
  match state {
    SLOT_WAITING => Waiting
    SLOT_PREFILL => Prefill
    SLOT_DECODE => Decode
    SLOT_FINISHING => Finishing
    SLOT_CANCELLED => Cancelled
    SLOT_DEADLINE => DeadlineExpired
    _ => Failed
  }
}
