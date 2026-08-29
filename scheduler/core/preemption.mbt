///|
fn Scheduler::aged_requires_preemption(
  self : Scheduler,
  urgent : Int,
) -> Bool raise SchedulerError {
  if urgent == NO_INDEX ||
    !self.is_aged(urgent) ||
    self.in_flight[urgent] ||
    (
      self.slot_states[urgent] != SLOT_WAITING &&
      self.slot_states[urgent] != SLOT_PREFILL
    ) {
    return false
  }
  let start = self.prefill_start(urgent)
  let target = self.prefill_target(urgent)
  let chunk = smaller(
    target - start,
    smaller(
      self.capacity.prefill_chunk_tokens(),
      self.capacity.step_token_budget(),
    ),
  )
  if chunk <= 0 || self.capacity.max_rows_per_step() <= 0 {
    return false
  }
  let old_pages = self.slot_page_count(urgent)
  let target_pages = ceiling_pages(
    start + chunk,
    self.capacity.tokens_per_page(),
  )
  let recipe_length = if self.recompute_required[urgent] ||
    start + chunk != target {
    self.intermediate_prefill_recipe.length()
  } else {
    self.final_prefill_recipe.length()
  }
  if target_pages > self.capacity.worker_max_plan_pages() ||
    recipe_length > self.capacity.worker_max_capabilities() {
    return false
  }
  let needed = target_pages - old_pages
  let free = self.evict_for_virtual_room(
    self.page_allocator.free_page_count(),
    needed + self.capacity.emergency_decode_page_reserve(),
  )
  let active_blocked = self.slot_states[urgent] == SLOT_WAITING &&
    self.active_count >= self.capacity.max_active_requests()
  let pages_blocked = needed > free ||
    free - needed < self.capacity.emergency_decode_page_reserve()
  active_blocked || pages_blocked
}

///|
fn Scheduler::preemption_victim(self : Scheduler, urgent : Int) -> Int {
  if urgent == NO_INDEX || self.active_count == 0 {
    return NO_INDEX
  }
  let start = self.active_cursor_position(self.preemption_cursor)
  for offset in 0..<self.active_count {
    let slot = self.active_slots[(start + offset) % self.active_count]
    if slot != urgent &&
      !self.in_flight[slot] &&
      (
        self.slot_states[slot] == SLOT_DECODE ||
        self.slot_states[slot] == SLOT_PREFILL
      ) {
      return slot
    }
  }
  NO_INDEX
}

///|
fn Scheduler::release_active_for_recompute(
  self : Scheduler,
  slot : Int,
) -> Unit raise SchedulerError {
  self.preflight_active_release(slot)
  let table = try! self.request_tables.read(slot)
  try! self.block_tables.reset_and_release(table, self.release_pages)
  for index in 0..<self.release_pages.length() {
    try! self.page_allocator.release_active(
      try! self.release_pages.page_at(index),
    )
  }
  try! self.request_tables.clear(slot)
  self.release_prefix_entry(slot)
  self.preemption_cursor = slot
  self.remove_active(slot)
  self.recompute_required[slot] = self.output_counts[slot] > 0
  self.recompute_processed[slot] = 0
  self.slot_states[slot] = SLOT_WAITING
  try! self.discover_prefix_candidate(self.requests[slot].unwrap(), slot)
  self.recompute_processed[slot] = 0
  self.prefill_cursors[slot] = 0
  self.admitted_steps[slot] = self.step
  self.enqueue_waiting(slot)
}

///|
fn Scheduler::preempt_for_aged_waiter(
  self : Scheduler,
) -> Bool raise SchedulerError {
  let urgent = self.oldest_prefill()
  if !self.aged_requires_preemption(urgent) {
    return false
  }
  let victim = self.preemption_victim(urgent)
  if victim == NO_INDEX {
    return false
  }
  self.release_active_for_recompute(victim)
  true
}
