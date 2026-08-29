///|
fn Scheduler::preflight_table_page_release(
  self : Scheduler,
  slot : Int,
) -> Unit raise SchedulerError {
  let table = self.request_tables.read(slot) catch {
    error => raise BlockTable(error)
  }
  let page_count = self.block_tables.page_count(table) catch {
    error => raise BlockTable(error)
  }
  if page_count > self.release_pages.capacity() {
    raise invalid(PageCapacity, OutOfRange)
  }
  for index in 0..<page_count {
    let page = self.block_tables.page_at(table, index) catch {
      error => raise BlockTable(error)
    }
    let references = self.page_allocator.active_reference_count(page) catch {
      error => raise PageAllocator(error)
    }
    if references <= 0 {
      raise invalid(PageCapacity, InvalidTransition)
    }
  }
}

///|
fn Scheduler::preflight_prefix_entry_release(
  self : Scheduler,
  slot : Int,
) -> Unit raise SchedulerError {
  let entry_index = self.prefix_entry_indices[slot]
  if entry_index == NO_INDEX {
    return
  }
  if !self.capacity.prefix_enabled() {
    raise invalid(RequestState, InvalidTransition)
  }
  let entry = @radix.PrefixEntryId::new(
    index=entry_index,
    generation=self.prefix_entry_generations[slot],
  ) catch {
    _ => raise invalid(RequestState, InvalidTransition)
  }
  let references = self.prefix_index.unwrap().active_reference_count(entry) catch {
      _ => raise invalid(RequestState, InvalidTransition)
    }
  if references <= 0 {
    raise invalid(RequestState, InvalidTransition)
  }
}

///|
fn Scheduler::preflight_waiting_prefix_device_loss(
  self : Scheduler,
) -> Unit raise SchedulerError {
  if !self.capacity.prefix_enabled() {
    return
  }
  for slot in 0..<self.slot_states.length() {
    if self.slot_states[slot] == SLOT_WAITING {
      let occupied = self.request_tables.is_occupied(slot) catch {
        error => raise BlockTable(error)
      }
      let has_entry = self.prefix_entry_indices[slot] != NO_INDEX
      if occupied || has_entry {
        raise invalid(RequestState, InvalidTransition)
      }
    }
  }
}

///|
fn Scheduler::release_waiting_prefix_device_state(self : Scheduler) -> Unit {
  if !self.capacity.prefix_enabled() {
    return
  }
  for slot in 0..<self.slot_states.length() {
    if self.slot_states[slot] == SLOT_WAITING {
      self.release_waiting_prefix(slot)
      self.prefill_cursors[slot] = 0
      self.recompute_processed[slot] = 0
    }
  }
}

///|
fn Scheduler::clear_cached_prefix_device_state(self : Scheduler) -> Unit {
  if !self.capacity.prefix_enabled() {
    return
  }
  let index = self.prefix_index.unwrap()
  while index.entry_count() > 0 {
    let before = index.entry_count()
    ignore(self.evict_one_prefix())
    if index.entry_count() == before {
      abort("scheduler prefix device-loss eviction invariant")
    }
  }
}
