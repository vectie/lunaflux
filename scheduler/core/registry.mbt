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
fn Scheduler::recycle_slot(self : Scheduler, index : Int) -> Unit {
  if self.prefix_entry_indices[index] != NO_INDEX ||
    (try! self.request_tables.is_occupied(index)) {
    abort("scheduler prefix recycle ownership invariant")
  }
  try! self.request_stop_tokens[index].release()
  self.requests[index] = None
  self.deadlines[index] = 0UL
  self.admitted_steps[index] = 0UL
  self.processed_inputs[index] = 0
  self.prefill_cursors[index] = 0
  self.output_counts[index] = 0
  self.in_flight[index] = false
  self.decode_tokens[index] = 0
  self.recompute_required[index] = false
  self.recompute_processed[index] = 0
  self.prefix_matched_tokens[index] = 0
  self.prefix_candidate_entry_indices[index] = NO_INDEX
  self.prefix_candidate_entry_generations[index] = 0
  self.prefix_candidate_tokens[index] = 0
  self.waiting_prev[index] = NO_INDEX
  self.waiting_next[index] = NO_INDEX
  self.waiting_link_generations[index] = 0
  self.live_count -= 1
  if self.slot_generations[index] == MAX_SLOT_GENERATION {
    self.slot_states[index] = SLOT_RETIRED
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
fn Scheduler::preflight_publications(
  self : Scheduler,
  count : Int,
) -> Unit raise SchedulerError {
  if count < 0 || self.publication_cursor > MAX_GENERATION - count.to_uint64() {
    raise invalid(PublicationSequence, Exhausted)
  }
}

///|
fn Scheduler::allocate_publication_sequence(
  self : Scheduler,
) -> UInt64 raise SchedulerError {
  self.preflight_publications(1)
  self.publication_cursor += 1UL
  self.publication_cursor
}

///|
fn Scheduler::enqueue_terminal(
  self : Scheduler,
  slot : Int,
  request_id : @inference.RequestId,
  generation : @inference.RequestGeneration,
  reason : TerminalReason,
) -> Unit raise SchedulerError {
  if self.terminal_available() == 0 {
    raise invalid(TerminalNoticeCapacity, Backpressured)
  }
  let sequence = self.allocate_publication_sequence()
  self.deadline_remove(slot)
  let index = self.terminal_tail
  self.terminal_slot_indices[index] = slot
  self.terminal_slot_generations[index] = self.slot_generations[slot]
  self.terminal_route_lanes[index] = self.publication_route_lanes[slot]
  self.terminal_route_generations[index] = self.publication_route_generations[slot]
  self.terminal_request_ids[index] = request_id
  self.terminal_generations[index] = generation
  self.terminal_reasons[index] = reason
  self.terminal_cached_input_tokens[index] = self.prefix_matched_tokens[slot]
  self.terminal_sequences[index] = sequence
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
  intermediate_prefill_recipe : @worker_protocol.RowCapabilityRecipe,
  final_prefill_recipe : @worker_protocol.RowCapabilityRecipe,
  decode_recipe : @worker_protocol.RowCapabilityRecipe,
) -> Scheduler raise SchedulerError {
  if intermediate_prefill_recipe.identity() != model_identity ||
    final_prefill_recipe.identity() != model_identity ||
    decode_recipe.identity() != model_identity ||
    intermediate_prefill_recipe.model_plan_generation() != model_plan_generation ||
    final_prefill_recipe.model_plan_generation() != model_plan_generation ||
    decode_recipe.model_plan_generation() != model_plan_generation {
    raise invalid(ModelIdentity, Mismatch)
  }
  let rows = capacity.max_rows_per_step()
  let prefill_max = if intermediate_prefill_recipe.length() >
    final_prefill_recipe.length() {
    intermediate_prefill_recipe.length()
  } else {
    final_prefill_recipe.length()
  }
  let max_recipe_length = if prefill_max > decode_recipe.length() {
    prefill_max
  } else {
    decode_recipe.length()
  }
  if rows > capacity.worker_max_capabilities() / max_recipe_length {
    raise invalid(Plan, OutOfRange)
  }
  let worker_limits = try
    @worker_protocol.WorkerProtocolLimits::new(
      max_prefill_rows=capacity.worker_max_prefill_rows(),
      max_decode_rows=capacity.worker_max_decode_rows(),
      max_plan_rows=capacity.worker_max_plan_rows(),
      max_plan_tokens=capacity.worker_max_plan_tokens(),
      max_plan_pages=capacity.worker_max_plan_pages(),
      max_capabilities=capacity.worker_max_capabilities(),
      max_completion_slots=capacity.worker_max_completion_slots(),
      max_sequence_tokens=capacity.worker_max_sequence_tokens(),
      max_token_id=capacity.worker_max_token_id(),
    )
  catch {
    error => raise WorkerProtocol(error)
  } noraise {
    value => value
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
  let request_stop_tokens = FixedArray::make(
    slots,
    @inference.LunaRequestStopTokenRetentionSlot::new(),
  )
  for index in 1..<slots {
    request_stop_tokens[index] = @inference.LunaRequestStopTokenRetentionSlot::new()
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
  let prefix_index = match capacity.prefix_index_limits() {
    Some(limits) => Some(try! @radix.PrefixIndex::new(limits))
    None => None
  }
  let prefix_tokens = match capacity.prefix_index_limits() {
    Some(limits) =>
      Some(try! @radix.PrefixTokenBuffer::new(limits.max_tokens_per_entry()))
    None => None
  }
  let prefix_pages = match capacity.prefix_index_limits() {
    Some(limits) =>
      Some(try! @radix.PrefixPublishBuffer::new(limits.page_capacity()))
    None => None
  }
  let prefix_lookup = match capacity.prefix_index_limits() {
    Some(limits) =>
      Some(try! @radix.PrefixLookupBuffer::new(limits.page_capacity()))
    None => None
  }
  let prefix_eviction = match capacity.prefix_index_limits() {
    Some(limits) =>
      Some(try! @radix.PrefixEvictionBuffer::new(limits.page_capacity()))
    None => None
  }
  let plan_a = try
    @worker_protocol.SchedulePlanBuffer::new(worker_limits)
  catch {
    error => raise WorkerProtocol(error)
  } noraise {
    owner => owner
  }
  let plan_b = try
    @worker_protocol.SchedulePlanBuffer::new(worker_limits)
  catch {
    error => raise WorkerProtocol(error)
  } noraise {
    owner => owner
  }
  let plan_buffer_binding = scheduler_plan_buffer_binding(
    plan_a.identity(),
    plan_b.identity(),
  )
  let completion_a = try
    @worker_protocol.CompletionBuffer::new(worker_limits)
  catch {
    error => raise WorkerProtocol(error)
  } noraise {
    owner => owner
  }
  let completion_b = try
    @worker_protocol.CompletionBuffer::new(worker_limits)
  catch {
    error => raise WorkerProtocol(error)
  } noraise {
    owner => owner
  }
  let first_completion = try
    @worker_protocol.CompletionSlot::new(0, worker_limits)
  catch {
    error => raise WorkerProtocol(error)
  } noraise {
    value => value
  }
  let completion_slots = FixedArray::make(rows, first_completion)
  for index in 1..<rows {
    completion_slots[index] = try
      @worker_protocol.CompletionSlot::new(index, worker_limits)
    catch {
      error => raise WorkerProtocol(error)
    } noraise {
      value => value
    }
  }
  let request_tables = try @block_table.BlockTableIdStorage::new(slots) catch {
    error => raise BlockTable(error)
  } noraise {
    storage => storage
  }
  let journal_tables = try @block_table.BlockTableIdStorage::new(rows) catch {
    error => raise BlockTable(error)
  } noraise {
    storage => storage
  }
  let page_run = try
    @page_allocator.PageRunBuffer::new(capacity.block_table_pages_per_request())
  catch {
    error => raise PageAllocator(error)
  } noraise {
    buffer => buffer
  }
  let release_pages = try
    @block_table.BlockTablePageBuffer::new(
      capacity.block_table_pages_per_request(),
    )
  catch {
    error => raise BlockTable(error)
  } noraise {
    buffer => buffer
  }
  {
    capacity,
    worker_limits,
    model_identity,
    page_allocator,
    block_tables,
    prefix_index,
    prefix_tokens,
    prefix_pages,
    prefix_lookup,
    prefix_eviction,
    model_plan_generation,
    intermediate_prefill_recipe,
    final_prefill_recipe,
    decode_recipe,
    completion_slots,
    plan_a,
    plan_b,
    plan_buffer_binding,
    completion_a,
    completion_b,
    request_tables,
    journal_tables,
    page_run,
    release_pages,
    slot_states: FixedArray::make(slots, SLOT_FREE),
    slot_generations: FixedArray::make(slots, 0),
    requests: FixedArray::make(slots, None),
    request_stop_tokens,
    request_generations: FixedArray::make(
      slots,
      @inference.RequestGeneration::initial(),
    ),
    deadlines: FixedArray::make(slots, 0UL),
    admitted_steps: FixedArray::make(slots, 0UL),
    processed_inputs: FixedArray::make(slots, 0),
    prefill_cursors: FixedArray::make(slots, 0),
    output_counts: FixedArray::make(slots, 0),
    in_flight: FixedArray::make(slots, false),
    decode_tokens: FixedArray::make(slots, 0),
    generated_history: FixedArray::make(
      capacity.required_recompute_history_cells(),
      0,
    ),
    recompute_required: FixedArray::make(slots, false),
    recompute_processed: FixedArray::make(slots, 0),
    prefix_entry_indices: FixedArray::make(slots, NO_INDEX),
    prefix_entry_generations: FixedArray::make(slots, 0),
    prefix_matched_tokens: FixedArray::make(slots, 0),
    prefix_candidate_entry_indices: FixedArray::make(slots, NO_INDEX),
    prefix_candidate_entry_generations: FixedArray::make(slots, 0),
    prefix_candidate_tokens: FixedArray::make(slots, 0),
    prefix_selection_pages: try! @page_allocator.PageIdStorage::new(
      capacity.worker_max_plan_pages(),
    ),
    prefix_selection_offsets: FixedArray::make(rows, 0),
    prefix_selection_counts: FixedArray::make(rows, 0),
    prefix_selection_entries: FixedArray::make(rows, NO_INDEX),
    prefix_selection_entry_generations: FixedArray::make(rows, 0),
    waiting_prev: FixedArray::make(slots, NO_INDEX),
    waiting_next: FixedArray::make(slots, NO_INDEX),
    waiting_link_generations: FixedArray::make(slots, 0),
    waiting_selection_heap: FixedArray::make(slots, NO_INDEX),
    waiting_selection_ranks: FixedArray::make(slots, 0),
    deadline_heap_slots: FixedArray::make(slots, NO_INDEX),
    deadline_heap_generations: FixedArray::make(slots, 0),
    deadline_heap_positions: FixedArray::make(slots, NO_INDEX),
    deadline_due_heap: FixedArray::make(slots, NO_INDEX),
    free_next,
    active_slots: FixedArray::make(capacity.max_active_requests(), NO_INDEX),
    active_positions: FixedArray::make(slots, NO_INDEX),
    prefill_slots: FixedArray::make(rows, NO_INDEX),
    prefill_tokens: FixedArray::make(rows, 0),
    decode_slots: FixedArray::make(rows, NO_INDEX),
    journal_slots: FixedArray::make(rows, NO_INDEX),
    journal_old_page_counts: FixedArray::make(rows, 0),
    journal_appended: FixedArray::make(rows, false),
    plan_a_slots: FixedArray::make(rows, NO_INDEX),
    plan_b_slots: FixedArray::make(rows, NO_INDEX),
    terminal_request_ids: FixedArray::make(slots, placeholder_id),
    terminal_generations: FixedArray::make(
      slots,
      @inference.RequestGeneration::initial(),
    ),
    terminal_reasons: FixedArray::make(slots, CancelledByCaller),
    terminal_cached_input_tokens: FixedArray::make(slots, 0),
    terminal_sequences: FixedArray::make(slots, 0UL),
    terminal_slot_indices: FixedArray::make(slots, NO_INDEX),
    terminal_slot_generations: FixedArray::make(slots, 0),
    terminal_route_lanes: FixedArray::make(slots, NO_INDEX),
    terminal_route_generations: FixedArray::make(slots, 0UL),
    output_request_ids: FixedArray::make(
      capacity.output_event_capacity(),
      placeholder_id,
    ),
    output_generations: FixedArray::make(
      capacity.output_event_capacity(),
      @inference.RequestGeneration::initial(),
    ),
    output_token_ids: FixedArray::make(capacity.output_event_capacity(), 0),
    output_positions: FixedArray::make(capacity.output_event_capacity(), 0),
    output_sequences: FixedArray::make(capacity.output_event_capacity(), 0UL),
    output_slot_indices: FixedArray::make(
      capacity.output_event_capacity(),
      NO_INDEX,
    ),
    output_slot_generations: FixedArray::make(
      capacity.output_event_capacity(),
      0,
    ),
    output_route_lanes: FixedArray::make(
      capacity.output_event_capacity(),
      NO_INDEX,
    ),
    output_route_generations: FixedArray::make(
      capacity.output_event_capacity(),
      0UL,
    ),
    publication_route_lanes: FixedArray::make(slots, NO_INDEX),
    publication_route_generations: FixedArray::make(slots, 0UL),
    free_head: 0,
    free_tail: slots - 1,
    waiting_head: NO_INDEX,
    waiting_tail: NO_INDEX,
    waiting_count: 0,
    active_count: 0,
    deadline_heap_count: 0,
    live_count: 0,
    terminal_head: 0,
    terminal_tail: 0,
    terminal_count: 0,
    output_head: 0,
    output_tail: 0,
    output_count: 0,
    publication_cursor: 0UL,
    generation_cursor: 0UL,
    step: 0UL,
    last_plan_sequence: 0UL,
    last_retired_plan_sequence: 0UL,
    telemetry_plan_sequence: 0UL,
    telemetry_plan_rows: 0,
    telemetry_plan_tokens: 0,
    plan_a_sequence: 0UL,
    plan_b_sequence: 0UL,
    prefill_count: 0,
    decode_count: 0,
    journal_count: 0,
    preemption_cursor: NO_INDEX,
    decode_cursor: NO_INDEX,
    pending_decode_cursor: NO_INDEX,
    waiting_selection_count: 0,
    waiting_unlink_link_touches: 0,
    selection_waiting_visits: 0,
    selection_active_visits: 0,
    prefix_selection_page_count: 0,
    deadline_live_visits: 0,
    deadline_due_pops: 0,
    cancel_reserved: false,
    exclusive_admission_reserved: false,
    cancel_slot: NO_INDEX,
    cancel_slot_generation: 0,
    cancel_request_id: placeholder_id,
    cancel_request_generation: @inference.RequestGeneration::initial(),
    cancel_publication_cursor: 0UL,
    cancel_step: 0UL,
    cancel_token_taken: false,
    prefix_lookups: 0UL,
    prefix_hits: 0UL,
    prefix_misses: 0UL,
    prefix_evictions: 0UL,
    prefix_tokens_reused: 0UL,
    prefix_tokens_computed: 0UL,
    prefix_publications: 0UL,
    prefix_candidate_discoveries: 0UL,
  }
}
