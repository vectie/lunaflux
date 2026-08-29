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
    self.journal_tables.write(self.journal_count, table) catch {
      error => raise BlockTable(error)
    }
    self.journal_slots[self.journal_count] = slot
    self.journal_old_page_counts[self.journal_count] = 0
    self.journal_appended[self.journal_count] = false
    self.journal_count += 1
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
    let mut journal = -1
    for index in 0..<self.journal_count {
      if self.journal_slots[index] == slot &&
        (try! self.journal_tables.read(index)) == table {
        journal = index
        break
      }
    }
    if journal < 0 {
      journal = self.journal_count
      self.journal_tables.write(journal, table) catch {
        error => raise BlockTable(error)
      }
      self.journal_slots[journal] = slot
      self.journal_count += 1
    }
    self.journal_old_page_counts[journal] = old_count
    self.journal_appended[journal] = false
    self.page_allocator.allocate_run(needed, self.page_run) catch {
      error => raise PageAllocator(error)
    }
    self.block_tables.append_run(table, self.page_run) catch {
      error => raise BlockTable(error)
    }
    self.journal_appended[journal] = true
  }
}

///|
fn Scheduler::stage_selected_prefix_pages(
  self : Scheduler,
  row : Int,
  table : @block_table.BlockTableId,
) -> Unit raise SchedulerError {
  let offset = self.prefix_selection_offsets[row]
  for index in 0..<self.prefix_selection_counts[row] {
    let page = try self.prefix_selection_pages.read(offset + index) catch {
      error => raise PageAllocator(error)
    } noraise {
      value => value
    }
    self.block_tables.append_page(table, page) catch {
      error => raise BlockTable(error)
    }
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
fn[T] rollback_row_draft(
  draft : @worker_protocol.PlanRowDraft,
  error : SchedulerError,
) -> T raise SchedulerError {
  try! draft.rollback()
  raise error
}

///|
fn[T] rollback_prefix_selection(
  scheduler : Scheduler,
  error : SchedulerError,
) -> T raise SchedulerError {
  scheduler.rollback_selected_prefixes()
  raise error
}

///|
fn[T] rollback_plan_build(
  scheduler : Scheduler,
  plan : @worker_protocol.SchedulePlanBuffer,
  plan_checkpoint : @worker_protocol.PlanBuildCheckpoint,
  table_checkpoint : @block_table.BlockTableAllocationCheckpoint,
  page_checkpoint : @page_allocator.PageAllocationCheckpoint,
  error : SchedulerError,
) -> T raise SchedulerError {
  try! plan.rollback_to(plan_checkpoint)
  scheduler.rollback_selected_prefixes()
  scheduler.rollback_tables(table_checkpoint)
  try! scheduler.block_tables.rollback_to(table_checkpoint)
  try! scheduler.page_allocator.rollback_to(page_checkpoint)
  raise error
}

///|
fn Scheduler::build_prefill_row(
  self : Scheduler,
  plan : @worker_protocol.SchedulePlanBuffer,
  row : Int,
) -> Unit raise SchedulerError {
  let slot = self.prefill_slots[row]
  let request = self.requests[slot].unwrap()
  let start = self.prefill_start(slot)
  let length = self.prefill_tokens[row]
  let table = self.row_table(slot)
  self.stage_selected_prefix_pages(row, table)
  let old_count = try self.block_tables.page_count(table) catch {
    error => raise BlockTable(error)
  } noraise {
    value => value
  }
  let target = ceiling_pages(start + length, self.capacity.tokens_per_page())
  self.append_pages(table, old_count, target, slot)
  let completion = self.completion_slots[row]
  let is_recompute = self.recompute_required[slot]
  let is_final = !is_recompute &&
    start + length == request.input_tokens().length()
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
    for index in start..<(start + length) {
      let token = self.prefill_token_at(slot, index)
      draft.push_token(token) catch {
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
    error => rollback_row_draft(draft, error)
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
    error => rollback_row_draft(draft, error)
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
  let a_reusable = self.plan_a_sequence == 0UL &&
    self.plan_a.state() == Writable &&
    self.plan_a.epoch() < MAX_GENERATION &&
    self.completion_a.state() == CompletionWritable &&
    self.completion_a.epoch() < MAX_GENERATION
  let b_reusable = self.plan_b_sequence == 0UL &&
    self.plan_b.state() == Writable &&
    self.plan_b.epoch() < MAX_GENERATION &&
    self.completion_b.state() == CompletionWritable &&
    self.completion_b.epoch() < MAX_GENERATION
  let side = if a_reusable {
    0
  } else if b_reusable {
    1
  } else if self.plan_a_sequence != 0UL || self.plan_b_sequence != 0UL {
    return BuildBackpressured
  } else {
    raise invalid(Plan, Exhausted)
  }
  let preempted = self.preempt_for_aged_waiter()
  if preempted {
    return BuildPreempted
  }
  let token_budget = self.select_rows() catch {
    error => rollback_prefix_selection(self, error)
  }
  if token_budget == 0 {
    return BuildIdle
  }
  let plan = if side == 0 { self.plan_a } else { self.plan_b }
  let sequence_value = self.last_plan_sequence + 1UL
  let has_previous = self.last_plan_sequence != 0UL
  let previous_sequence_value = self.last_plan_sequence
  let table_checkpoint = try self.block_tables.checkpoint() catch {
    error => {
      self.rollback_selected_prefixes()
      raise BlockTable(error)
    }
  } noraise {
    value => value
  }
  let page_checkpoint = try self.page_allocator.checkpoint() catch {
    error => {
      self.rollback_selected_prefixes()
      try! self.block_tables.rollback_to(table_checkpoint)
      raise PageAllocator(error)
    }
  } noraise {
    value => value
  }
  let plan_checkpoint = try plan.checkpoint() catch {
    error => {
      self.rollback_selected_prefixes()
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
    error =>
      rollback_plan_build(
        self, plan, plan_checkpoint, table_checkpoint, page_checkpoint, error,
      )
  }
  self.commit_plan_state(side)
  self.commit_selected_prefixes()
  self.last_plan_sequence = sequence_value
  self.telemetry_plan_sequence = sequence_value
  self.telemetry_plan_rows = self.prefill_count + self.decode_count
  self.telemetry_plan_tokens = token_budget
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
