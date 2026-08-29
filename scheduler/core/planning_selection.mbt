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
  if self.prefix_selection_page_count != 0 {
    abort("scheduler prefix selection cleanup invariant")
  }
  for index in 0..<self.prefill_count {
    self.prefill_slots[index] = NO_INDEX
    self.prefill_tokens[index] = 0
    self.prefix_selection_offsets[index] = 0
    self.prefix_selection_counts[index] = 0
    self.prefix_selection_entries[index] = NO_INDEX
    self.prefix_selection_entry_generations[index] = 0
  }
  for index in 0..<self.decode_count {
    self.decode_slots[index] = NO_INDEX
  }
  self.prefill_count = 0
  self.decode_count = 0
  self.journal_count = 0
  self.pending_decode_cursor = self.decode_cursor
  self.waiting_selection_count = 0
  self.selection_waiting_visits = 0
  self.selection_active_visits = 0
}

///|
fn Scheduler::table_page_count(
  self : Scheduler,
  slot : Int,
) -> Int raise SchedulerError {
  let table = self.request_tables.read(slot) catch {
    error => raise BlockTable(error)
  }
  self.block_tables.page_count(table) catch {
    error => raise BlockTable(error)
  }
}

///|
fn Scheduler::slot_page_count(
  self : Scheduler,
  slot : Int,
) -> Int raise SchedulerError {
  let occupied = try self.request_tables.is_occupied(slot) catch {
    error => raise BlockTable(error)
  } noraise {
    value => value
  }
  if occupied {
    self.table_page_count(slot)
  } else {
    0
  }
}

///|
fn Scheduler::oldest_prefill(self : Scheduler) -> Int {
  let mut oldest = self.waiting_head
  for index in 0..<self.active_count {
    let slot = self.active_slots[index]
    self.selection_active_visits += 1
    if self.slot_states[slot] == SLOT_PREFILL &&
      !self.in_flight[slot] &&
      (
        oldest == NO_INDEX ||
        self.admitted_steps[slot] < self.admitted_steps[oldest]
      ) {
      oldest = slot
    }
  }
  oldest
}

///|
fn Scheduler::is_aged(self : Scheduler, slot : Int) -> Bool {
  slot != NO_INDEX &&
  self.step >= self.admitted_steps[slot] &&
  self.step - self.admitted_steps[slot] >=
  self.capacity.waiting_age_threshold_steps().to_uint64()
}

///|
fn Scheduler::select_rows(self : Scheduler) -> Int raise SchedulerError {
  self.clear_selection()
  let row_limit = self.capacity.max_rows_per_step()
  let mut token_left = self.capacity.step_token_budget()
  let mut virtual_free = self.page_allocator.free_page_count()
  let mut plan_pages = 0
  let mut plan_capabilities = 0
  let mut active_room = self.capacity.max_active_requests() - self.active_count
  let oldest = self.oldest_prefill()
  let aged = self.is_aged(oldest)
  let mut aged_selected = NO_INDEX

  // An aged row is debited only when it is exactly eligible. An ineligible
  // aged prompt never reserves tokens or pages from runnable decodes.
  if aged &&
    row_limit > 0 &&
    token_left > 0 &&
    !self.in_flight[oldest] &&
    (
      self.slot_states[oldest] == SLOT_PREFILL ||
      (self.slot_states[oldest] == SLOT_WAITING && active_room > 0)
    ) {
    let waiting = self.slot_states[oldest] == SLOT_WAITING
    let matched = if waiting {
      self.revalidate_waiting_prefix(oldest)
    } else {
      0
    }
    let start = if waiting { matched } else { self.prefill_start(oldest) }
    let target = self.prefill_target(oldest)
    let chunk = smaller(
      target - start,
      smaller(self.capacity.prefill_chunk_tokens(), token_left),
    )
    let old_pages = if waiting {
      matched / self.capacity.tokens_per_page()
    } else {
      self.slot_page_count(oldest)
    }
    let target_pages = if start + chunk == 0 {
      0
    } else {
      ceiling_pages(start + chunk, self.capacity.tokens_per_page())
    }
    let needed = target_pages - old_pages
    let recipe_length = if self.recompute_required[oldest] ||
      start + chunk != target {
      self.intermediate_prefill_recipe.length()
    } else {
      self.final_prefill_recipe.length()
    }
    if chunk > 0 &&
      plan_pages + target_pages <= self.capacity.worker_max_plan_pages() &&
      plan_capabilities + recipe_length <=
      self.capacity.worker_max_capabilities() {
      virtual_free = self.evict_for_virtual_room(
        virtual_free,
        needed + self.capacity.emergency_decode_page_reserve(),
      )
      if needed <= virtual_free &&
        virtual_free - needed >= self.capacity.emergency_decode_page_reserve() {
        self.prefill_slots[0] = oldest
        self.prefill_tokens[0] = chunk
        if waiting {
          self.acquire_revalidated_prefix(oldest, 0, matched)
        }
        self.prefill_count = 1
        token_left -= chunk
        virtual_free -= needed
        plan_pages += target_pages
        plan_capabilities += recipe_length
        aged_selected = oldest
        if self.slot_states[oldest] == SLOT_WAITING {
          active_room -= 1
        }
      }
    }
  }

  // Decode candidates use a persistent slot cursor. Selection only stages the
  // next cursor; commit_plan_state publishes it after the plan is submitted.
  let active_count = self.active_count
  let decode_start = self.active_cursor_position(self.decode_cursor)
  let mut scanned = 0
  while scanned < active_count &&
        self.prefill_count + self.decode_count < row_limit &&
        token_left > 0 {
    let position = (decode_start + scanned) % active_count
    let slot = self.active_slots[position]
    scanned += 1
    self.selection_active_visits += 1
    if self.slot_states[slot] == SLOT_DECODE && !self.in_flight[slot] {
      let old_pages = self.table_page_count(slot)
      let request = self.requests[slot].unwrap()
      let kv_tokens = request.input_tokens().length() + self.output_counts[slot]
      let target_pages = ceiling_pages(
        kv_tokens,
        self.capacity.tokens_per_page(),
      )
      let needed = target_pages - old_pages
      if plan_pages + target_pages <= self.capacity.worker_max_plan_pages() &&
        plan_capabilities + self.decode_recipe.length() <=
        self.capacity.worker_max_capabilities() {
        virtual_free = self.evict_for_virtual_room(virtual_free, needed)
      }
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
        self.pending_decode_cursor = self.active_slots[(position + 1) %
          active_count]
      }
    }
  }

  // An aged pass remains FIFO. Otherwise one bottom-up heap preserves prefix
  // depth priority with FIFO tie-breaking without rescanning the queue.
  let prefix_ordered = !aged
  if prefix_ordered {
    self.prepare_waiting_selection()
  }
  let mut fifo_current = self.waiting_head
  for _ in 0..<self.waiting_count {
    if self.prefill_count + self.decode_count >= row_limit || token_left == 0 {
      break
    }
    let current = if prefix_ordered {
      self.pop_waiting_selection()
    } else {
      let selected = fifo_current
      if selected != NO_INDEX {
        fifo_current = self.waiting_successor(selected)
        self.selection_waiting_visits += 1
      }
      selected
    }
    if current == NO_INDEX {
      break
    }
    if current != aged_selected && !self.in_flight[current] && active_room > 0 {
      let matched = self.revalidate_waiting_prefix(current)
      let start = matched
      let target = self.prefill_target(current)
      let chunk = smaller(
        target - start,
        smaller(self.capacity.prefill_chunk_tokens(), token_left),
      )
      let target_pages = if start + chunk == 0 {
        0
      } else {
        ceiling_pages(start + chunk, self.capacity.tokens_per_page())
      }
      let needed = target_pages - matched / self.capacity.tokens_per_page()
      let recipe_length = if self.recompute_required[current] ||
        start + chunk != target {
        self.intermediate_prefill_recipe.length()
      } else {
        self.final_prefill_recipe.length()
      }
      if chunk > 0 &&
        plan_pages + target_pages <= self.capacity.worker_max_plan_pages() &&
        plan_capabilities + recipe_length <=
        self.capacity.worker_max_capabilities() {
        virtual_free = self.evict_for_virtual_room(
          virtual_free,
          needed + self.capacity.emergency_decode_page_reserve(),
        )
        if needed <= virtual_free &&
          virtual_free - needed >= self.capacity.emergency_decode_page_reserve() {
          self.prefill_slots[self.prefill_count] = current
          self.prefill_tokens[self.prefill_count] = chunk
          self.acquire_revalidated_prefix(current, self.prefill_count, matched)
          self.prefill_count += 1
          token_left -= chunk
          virtual_free -= needed
          plan_pages += target_pages
          plan_capabilities += recipe_length
          active_room -= 1
        }
      }
    }
  }

  // Active prefills share the remainder in current deterministic array order.
  for index in 0..<self.active_count {
    let slot = self.active_slots[index]
    self.selection_active_visits += 1
    if self.prefill_count + self.decode_count >= row_limit || token_left == 0 {
      break
    }
    if slot != aged_selected &&
      self.slot_states[slot] == SLOT_PREFILL &&
      !self.in_flight[slot] {
      let start = self.prefill_start(slot)
      let target = self.prefill_target(slot)
      let chunk = smaller(
        target - start,
        smaller(self.capacity.prefill_chunk_tokens(), token_left),
      )
      let old_pages = self.table_page_count(slot)
      let target_pages = ceiling_pages(
        start + chunk,
        self.capacity.tokens_per_page(),
      )
      let needed = target_pages - old_pages
      let recipe_length = if self.recompute_required[slot] ||
        start + chunk != target {
        self.intermediate_prefill_recipe.length()
      } else {
        self.final_prefill_recipe.length()
      }
      if chunk > 0 &&
        plan_pages + target_pages <= self.capacity.worker_max_plan_pages() &&
        plan_capabilities + recipe_length <=
        self.capacity.worker_max_capabilities() {
        virtual_free = self.evict_for_virtual_room(
          virtual_free,
          needed + self.capacity.emergency_decode_page_reserve(),
        )
        if needed <= virtual_free &&
          virtual_free - needed >= self.capacity.emergency_decode_page_reserve() {
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
  }
  if self.selection_waiting_visits > self.waiting_selection_heap.length() ||
    self.selection_active_visits > self.active_slots.length() * 3 {
    abort("scheduler selection complexity invariant")
  }
  self.capacity.step_token_budget() - token_left
}
