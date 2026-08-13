///|
fn smaller(left : Int, right : Int) -> Int {
  if left < right {
    left
  } else {
    right
  }
}

///|
fn Scheduler::clear_selection(self : Scheduler) -> Unit {
  self.prefill_count = 0
  self.decode_count = 0
  self.journal_count = 0
}

///|
fn Scheduler::table_page_count(
  self : Scheduler,
  slot : Int,
) -> Int raise SchedulerError {
  let table = try self.request_tables.read(slot) catch {
    error => raise BlockTable(error)
  } noraise {
    value => value
  }
  try self.block_tables.page_count(table) catch {
    error => raise BlockTable(error)
  } noraise {
    value => value
  }
}

///|
fn Scheduler::select_rows(self : Scheduler) -> Int raise SchedulerError {
  self.clear_selection()
  let row_limit = self.capacity.max_rows_per_step()
  let mut token_left = self.capacity.step_token_budget()
  let mut virtual_free = self.page_allocator.free_page_count()
  let mut plan_pages = 0
  let mut plan_capabilities = 0
  let mut oldest_prefill = self.waiting_head
  for active_index in 0..<self.active_count {
    let slot = self.active_slots[active_index]
    if self.slot_states[slot] == SLOT_PREFILL &&
      !self.in_flight[slot] &&
      (
        oldest_prefill == NO_INDEX ||
        self.admitted_steps[slot] < self.admitted_steps[oldest_prefill]
      ) {
      oldest_prefill = slot
    }
  }
  let aged = oldest_prefill != NO_INDEX &&
    self.step - self.admitted_steps[oldest_prefill] >=
    self.capacity.waiting_age_threshold_steps().to_uint64()
  let reserved_prefill = if aged {
    let request = self.requests[oldest_prefill].unwrap()
    let remaining = request.input_tokens().length() -
      self.processed_inputs[oldest_prefill]
    smaller(remaining, self.capacity.prefill_chunk_tokens())
  } else {
    0
  }
  for active_index in 0..<self.active_count {
    let slot = self.active_slots[active_index]
    if self.slot_states[slot] == SLOT_DECODE &&
      !self.in_flight[slot] &&
      self.decode_count + self.prefill_count < row_limit &&
      token_left > reserved_prefill {
      let old_pages = self.table_page_count(slot)
      let request = self.requests[slot].unwrap()
      let kv_tokens = request.input_tokens().length() + self.output_counts[slot]
      let target_pages = ceiling_pages(
        kv_tokens,
        self.capacity.tokens_per_page(),
      )
      let needed = target_pages - old_pages
      if needed <= virtual_free &&
        plan_pages + target_pages <= self.capacity.worker_max_plan_pages() &&
        plan_capabilities + self.decode_recipe.length() <=
        self.capacity.worker_max_capabilities() {
        self.decode_slots[self.decode_count] = slot
        self.decode_count += 1
        token_left -= 1
        virtual_free -= needed
        plan_pages += target_pages
        plan_capabilities += self.decode_recipe.length()
      }
    }
  }
  let mut current = if aged { oldest_prefill } else { self.waiting_head }
  let mut active_room = self.capacity.max_active_requests() - self.active_count
  while current != NO_INDEX &&
        self.prefill_count + self.decode_count < row_limit &&
        token_left > 0 {
    let next = if self.slot_states[current] == SLOT_WAITING {
      self.waiting_next[current]
    } else if aged && current == oldest_prefill {
      self.waiting_head
    } else {
      NO_INDEX
    }
    if !self.in_flight[current] &&
      (
        self.slot_states[current] == SLOT_PREFILL ||
        (self.slot_states[current] == SLOT_WAITING && active_room > 0)
      ) {
      let request = self.requests[current].unwrap()
      let remaining = request.input_tokens().length() -
        self.processed_inputs[current]
      let chunk = smaller(
        remaining,
        smaller(self.capacity.prefill_chunk_tokens(), token_left),
      )
      let old_pages = if self.slot_states[current] == SLOT_WAITING {
        0
      } else {
        self.table_page_count(current)
      }
      let target_pages = ceiling_pages(
        self.processed_inputs[current] + chunk,
        self.capacity.tokens_per_page(),
      )
      let needed = target_pages - old_pages
      let recipe_length = if self.processed_inputs[current] + chunk ==
        request.input_tokens().length() {
        self.final_prefill_recipe.length()
      } else {
        self.intermediate_prefill_recipe.length()
      }
      if chunk > 0 &&
        virtual_free - needed >= self.capacity.emergency_decode_page_reserve() &&
        plan_pages + target_pages <= self.capacity.worker_max_plan_pages() &&
        plan_capabilities + recipe_length <=
        self.capacity.worker_max_capabilities() {
        self.prefill_slots[self.prefill_count] = current
        self.prefill_tokens[self.prefill_count] = chunk
        self.prefill_count += 1
        token_left -= chunk
        virtual_free -= needed
        plan_pages += target_pages
        plan_capabilities += recipe_length
        if self.slot_states[current] == SLOT_WAITING {
          active_room -= 1
        }
      }
    }
    current = next
  }
  for active_index in 0..<self.active_count {
    let slot = self.active_slots[active_index]
    if self.prefill_count + self.decode_count >= row_limit || token_left == 0 {
      break
    }
    if self.slot_states[slot] == SLOT_PREFILL &&
      (!aged || slot != oldest_prefill) &&
      !self.in_flight[slot] {
      let request = self.requests[slot].unwrap()
      let remaining = request.input_tokens().length() -
        self.processed_inputs[slot]
      let chunk = smaller(
        remaining,
        smaller(self.capacity.prefill_chunk_tokens(), token_left),
      )
      let old_pages = self.table_page_count(slot)
      let target_pages = ceiling_pages(
        self.processed_inputs[slot] + chunk,
        self.capacity.tokens_per_page(),
      )
      let needed = target_pages - old_pages
      let recipe_length = if self.processed_inputs[slot] + chunk ==
        request.input_tokens().length() {
        self.final_prefill_recipe.length()
      } else {
        self.intermediate_prefill_recipe.length()
      }
      if chunk > 0 &&
        virtual_free - needed >= self.capacity.emergency_decode_page_reserve() &&
        plan_pages + target_pages <= self.capacity.worker_max_plan_pages() &&
        plan_capabilities + recipe_length <=
        self.capacity.worker_max_capabilities() {
        self.prefill_slots[self.prefill_count] = slot
        self.prefill_tokens[self.prefill_count] = chunk
        self.prefill_count += 1
        token_left -= chunk
        virtual_free -= needed
        plan_pages += target_pages
        plan_capabilities += recipe_length
      }
    }
  }
  self.capacity.step_token_budget() - token_left
}

///|
fn Scheduler::row_table(
  self : Scheduler,
  slot : Int,
) -> @block_table.BlockTableId raise SchedulerError {
  let occupied = try self.request_tables.is_occupied(slot) catch {
    error => raise BlockTable(error)
  } noraise {
    value => value
  }
  if occupied {
    let table = try self.request_tables.read(slot) catch {
      error => raise BlockTable(error)
    } noraise {
      value => value
    }
    table
  } else {
    let table = try self.block_tables.allocate() catch {
      error => raise BlockTable(error)
    } noraise {
      value => value
    }
    table
  }
}

///|
fn Scheduler::append_pages(
  self : Scheduler,
  table : @block_table.BlockTableId,
  old_count : Int,
  target_count : Int,
  slot : Int,
) -> Unit raise SchedulerError {
  let needed = target_count - old_count
  if needed > 0 {
    self.journal_tables.write(self.journal_count, table) catch {
      error => raise BlockTable(error)
    }
    self.journal_slots[self.journal_count] = slot
    self.journal_old_page_counts[self.journal_count] = old_count
    self.journal_appended[self.journal_count] = false
    self.journal_count += 1
    self.page_allocator.allocate_run(needed, self.page_run) catch {
      error => raise PageAllocator(error)
    }
    self.block_tables.append_run(table, self.page_run) catch {
      error => raise BlockTable(error)
    }
    self.journal_appended[self.journal_count - 1] = true
  }
}

///|
fn Scheduler::push_table(
  self : Scheduler,
  draft : @worker_protocol.PlanRowDraft,
  table : @block_table.BlockTableId,
) -> Unit raise SchedulerError {
  let count = try self.block_tables.page_count(table) catch {
    error => raise BlockTable(error)
  } noraise {
    value => value
  }
  for index in 0..<count {
    let page = try self.block_tables.page_at(table, index) catch {
      error => raise BlockTable(error)
    } noraise {
      value => value
    }
    self.page_allocator.validate_active(page) catch {
      error => raise PageAllocator(error)
    }
    draft.push_page(page) catch {
      error => raise WorkerProtocol(error)
    }
  }
}

///|
fn Scheduler::build_prefill_row(
  self : Scheduler,
  plan : @worker_protocol.SchedulePlanBuffer,
  row : Int,
) -> Unit raise SchedulerError {
  let slot = self.prefill_slots[row]
  let request = self.requests[slot].unwrap()
  let start = self.processed_inputs[slot]
  let length = self.prefill_tokens[row]
  let table = self.row_table(slot)
  let old_count = try self.block_tables.page_count(table) catch {
    error => raise BlockTable(error)
  } noraise {
    value => value
  }
  let target = ceiling_pages(start + length, self.capacity.tokens_per_page())
  self.append_pages(table, old_count, target, slot)
  let completion = self.completion_slots[row]
  let is_final = start + length == request.input_tokens().length()
  let draft = try {
    if is_final {
      plan.begin_final_prefill(
        request_id=request.request_id(),
        request_generation=self.request_generations[slot],
        sequence_token_start=start,
        sampling=request.sampling(),
        sampling_seed=request.sampling_seed(),
        sample_index=self.output_counts[slot],
        completion_slot=completion,
      )
    } else {
      plan.begin_prefill(
        request_id=request.request_id(),
        request_generation=self.request_generations[slot],
        sequence_token_start=start,
        completion_slot=completion,
      )
    }
  } catch {
    error => raise WorkerProtocol(error)
  } noraise {
    value => value
  }
  try {
    let tokens = request.input_tokens().token_ids()
    for index in start..<(start + length) {
      draft.push_token(tokens[index]) catch {
        error => raise WorkerProtocol(error)
      }
    }
    self.push_table(draft, table)
    let recipe = if is_final {
      self.final_prefill_recipe
    } else {
      self.intermediate_prefill_recipe
    }
    draft.push_recipe(recipe) catch {
      error => raise WorkerProtocol(error)
    }
    draft.commit() catch {
      error => raise WorkerProtocol(error)
    }
  } catch {
    error => {
      try! draft.rollback()
      raise error
    }
  }
}

///|
fn Scheduler::build_decode_row(
  self : Scheduler,
  plan : @worker_protocol.SchedulePlanBuffer,
  row : Int,
) -> Unit raise SchedulerError {
  let slot = self.decode_slots[row]
  let request = self.requests[slot].unwrap()
  let table = self.row_table(slot)
  let old_count = try self.block_tables.page_count(table) catch {
    error => raise BlockTable(error)
  } noraise {
    value => value
  }
  let position = request.input_tokens().length() + self.output_counts[slot] - 1
  let target = ceiling_pages(position + 1, self.capacity.tokens_per_page())
  self.append_pages(table, old_count, target, slot)
  let completion_index = self.prefill_count + row
  let completion = self.completion_slots[completion_index]
  let draft = try
    plan.begin_decode(
      request_id=request.request_id(),
      request_generation=self.request_generations[slot],
      sequence_token_position=position,
      sampling=request.sampling(),
      sampling_seed=request.sampling_seed(),
      sample_index=self.output_counts[slot],
      completion_slot=completion,
    )
  catch {
    error => raise WorkerProtocol(error)
  } noraise {
    value => value
  }
  try {
    draft.push_token(self.decode_tokens[slot]) catch {
      error => raise WorkerProtocol(error)
    }
    self.push_table(draft, table)
    draft.push_recipe(self.decode_recipe) catch {
      error => raise WorkerProtocol(error)
    }
    draft.commit() catch {
      error => raise WorkerProtocol(error)
    }
  } catch {
    error => {
      try! draft.rollback()
      raise error
    }
  }
}

///|
/// Builds one authenticated plan from fixed startup storage. At most two plans
/// may remain submitted until ordered completion retirement frees a side.
pub fn Scheduler::build_next(
  self : Scheduler,
) -> BuildNextOutcome raise SchedulerError {
  self.require_no_cancel_reservation()
  if self.step == MAX_GENERATION || self.last_plan_sequence == MAX_GENERATION {
    raise invalid(Plan, Exhausted)
  }
  let side = if self.plan_a_sequence == 0UL && self.plan_a.state() == Writable {
    0
  } else if self.plan_b_sequence == 0UL && self.plan_b.state() == Writable {
    1
  } else {
    return BuildBackpressured
  }
  let token_budget = self.select_rows()
  if token_budget == 0 {
    return BuildIdle
  }
  let plan = if side == 0 { self.plan_a } else { self.plan_b }
  let sequence_value = self.last_plan_sequence + 1UL
  let has_previous = self.last_plan_sequence != 0UL
  let previous_sequence_value = self.last_plan_sequence
  let table_checkpoint = try self.block_tables.checkpoint() catch {
    error => raise BlockTable(error)
  } noraise {
    value => value
  }
  let page_checkpoint = try self.page_allocator.checkpoint() catch {
    error => {
      try! self.block_tables.rollback_to(table_checkpoint)
      raise PageAllocator(error)
    }
  } noraise {
    value => value
  }
  let plan_checkpoint = try plan.checkpoint() catch {
    error => {
      try! self.block_tables.rollback_to(table_checkpoint)
      try! self.page_allocator.rollback_to(page_checkpoint)
      raise WorkerProtocol(error)
    }
  } noraise {
    value => value
  }
  let submitted = try {
    for row in 0..<self.prefill_count {
      self.build_prefill_row(plan, row)
    }
    for row in 0..<self.decode_count {
      self.build_decode_row(plan, row)
    }
    plan.submit_wire(
      sequence_value~,
      model_plan_generation_value=self.model_plan_generation.value(),
      token_budget~,
      has_previous~,
      previous_sequence_value~,
      loaded_generation_value=self.model_plan_generation.value(),
    ) catch {
      error => raise WorkerProtocol(error)
    }
  } catch {
    error => {
      try! plan.rollback_to(plan_checkpoint)
      self.rollback_tables(table_checkpoint)
      try! self.block_tables.rollback_to(table_checkpoint)
      try! self.page_allocator.rollback_to(page_checkpoint)
      raise error
    }
  } noraise {
    value => value
  }
  self.commit_plan_state(side)
  self.last_plan_sequence = sequence_value
  if side == 0 {
    self.plan_a_sequence = sequence_value
  } else {
    self.plan_b_sequence = sequence_value
  }
  self.step += 1UL
  try! self.block_tables.commit_checkpoint(table_checkpoint)
  try! self.page_allocator.commit_checkpoint(page_checkpoint)
  ignore(submitted)
  BuildSubmitted(side, sequence_value)
}
