///|
fn Scheduler::prefix_saturating_add(current : UInt64, amount : Int) -> UInt64 {
  let added = amount.to_uint64()
  if current > 0xffffffffffffffffUL - added {
    0xffffffffffffffffUL
  } else {
    current + added
  }
}

///|
fn Scheduler::copy_prefix_tokens(
  self : Scheduler,
  request : TokenizedRequest,
) -> Int {
  let buffer = self.prefix_tokens.unwrap()
  let limit = self.prefix_index.unwrap().limits().max_tokens_per_entry()
  let count = Int::min(request.input_tokens().length(), limit)
  buffer.clear()
  for index in 0..<count {
    try! buffer.push_token(try! request.input_token_at(index))
  }
  count
}

///|
fn Scheduler::prefix_hit_preflight(
  self : Scheduler,
  entry : @radix.PrefixEntryId,
  lookup : @radix.PrefixLookupBuffer,
) -> Bool {
  if self.block_tables.free_table_count() == 0 ||
    lookup.page_count() > self.capacity.block_table_pages_per_request() ||
    (try! self.prefix_index.unwrap().active_reference_count(entry)) >=
    self.prefix_index.unwrap().limits().max_active_references_per_entry() {
    return false
  }
  let maximum = self.page_allocator.limits().max_active_references_per_page()
  for index in 0..<lookup.page_count() {
    let page = try! lookup.page_at(index)
    if (try! self.page_allocator.active_reference_count(page)) >= maximum {
      return false
    }
  }
  true
}

///|
fn Scheduler::clear_prefix_candidate(self : Scheduler, slot : Int) -> Unit {
  self.prefix_candidate_entry_indices[slot] = NO_INDEX
  self.prefix_candidate_entry_generations[slot] = 0
  self.prefix_candidate_tokens[slot] = 0
}

///|
fn Scheduler::discover_prefix_candidate(
  self : Scheduler,
  request : TokenizedRequest,
  slot : Int,
) -> Unit raise SchedulerError {
  self.prefix_entry_indices[slot] = NO_INDEX
  self.prefix_entry_generations[slot] = 0
  self.prefix_matched_tokens[slot] = 0
  self.clear_prefix_candidate(slot)
  if !self.capacity.prefix_enabled() {
    return
  }
  if !request.has_prefix_key() {
    return
  }
  let key = request.prefix_key(self.capacity.prefix_layout())
  let permission = key.cache_permission() catch {
    _ => raise invalid(Request, Stale)
  }
  if permission == Disabled {
    return
  }
  let copied = self.copy_prefix_tokens(request)
  let maximum = Int::min(copied, request.input_tokens().length() - 1)
  self.prefix_lookups = Scheduler::prefix_saturating_add(self.prefix_lookups, 1)
  if maximum <= 0 {
    self.prefix_misses = Scheduler::prefix_saturating_add(self.prefix_misses, 1)
    return
  }
  let lookup = self.prefix_lookup.unwrap()
  self.prefix_index
  .unwrap()
  .lookup(
    key,
    self.prefix_tokens.unwrap(),
    maximum_reusable_tokens=maximum,
    lookup,
  ) catch {
    StaleKey => raise invalid(Request, Stale)
    _ => abort("scheduler prefix lookup invariant")
  }
  if !lookup.has_entry() {
    self.prefix_misses = Scheduler::prefix_saturating_add(self.prefix_misses, 1)
    return
  }
  let entry = try! lookup.entry_id()
  self.prefix_candidate_entry_indices[slot] = entry.index()
  self.prefix_candidate_entry_generations[slot] = entry.generation()
  self.prefix_candidate_tokens[slot] = lookup.matched_tokens()
  self.prefix_candidate_discoveries = Scheduler::prefix_saturating_add(
    self.prefix_candidate_discoveries,
    1,
  )
}

///|
fn Scheduler::revalidate_waiting_prefix(self : Scheduler, slot : Int) -> Int {
  let candidate = self.prefix_candidate_entry_indices[slot]
  if candidate == NO_INDEX {
    return 0
  }
  let request = self.requests[slot].unwrap()
  let key = try! request.prefix_key(self.capacity.prefix_layout())
  let copied = self.copy_prefix_tokens(request)
  let maximum = Int::min(copied, request.input_tokens().length() - 1)
  let lookup = self.prefix_lookup.unwrap()
  try! self.prefix_index
  .unwrap()
  .lookup(
    key,
    self.prefix_tokens.unwrap(),
    maximum_reusable_tokens=maximum,
    lookup,
  )
  let valid = lookup.has_entry() &&
    (try! lookup.entry_id()).index() == candidate &&
    (try! lookup.entry_id()).generation() ==
    self.prefix_candidate_entry_generations[slot] &&
    lookup.matched_tokens() == self.prefix_candidate_tokens[slot] &&
    self.prefix_hit_preflight(try! lookup.entry_id(), lookup)
  if !valid {
    self.clear_prefix_candidate(slot)
    self.prefix_misses = Scheduler::prefix_saturating_add(self.prefix_misses, 1)
    return 0
  }
  lookup.matched_tokens()
}

///|
fn Scheduler::acquire_revalidated_prefix(
  self : Scheduler,
  slot : Int,
  row : Int,
  matched : Int,
) -> Unit {
  if matched == 0 {
    self.prefix_selection_offsets[row] = self.prefix_selection_page_count
    self.prefix_selection_counts[row] = 0
    self.prefix_selection_entries[row] = NO_INDEX
    self.prefix_selection_entry_generations[row] = 0
    return
  }
  let lookup = self.prefix_lookup.unwrap()
  let entry = try! lookup.entry_id()
  if entry.index() != self.prefix_candidate_entry_indices[slot] ||
    entry.generation() != self.prefix_candidate_entry_generations[slot] ||
    lookup.matched_tokens() != matched {
    abort("scheduler prefix revalidation invariant")
  }
  let offset = self.prefix_selection_page_count
  if offset + lookup.page_count() > self.prefix_selection_pages.capacity() {
    abort("scheduler prefix selection page capacity invariant")
  }
  try! self.prefix_index.unwrap().retain_active(entry)
  for index in 0..<lookup.page_count() {
    let page = try! lookup.page_at(index)
    try! self.page_allocator.retain_active(page)
    try! self.prefix_selection_pages.write(offset + index, page)
  }
  self.prefix_selection_offsets[row] = offset
  self.prefix_selection_counts[row] = lookup.page_count()
  self.prefix_selection_entries[row] = entry.index()
  self.prefix_selection_entry_generations[row] = entry.generation()
  self.prefix_selection_page_count += lookup.page_count()
  self.prefix_entry_indices[slot] = entry.index()
  self.prefix_entry_generations[slot] = entry.generation()
  self.prefix_matched_tokens[slot] = matched
  if self.recompute_required[slot] {
    self.recompute_processed[slot] = matched
  } else {
    self.prefill_cursors[slot] = matched
  }
}

///|
fn Scheduler::clear_prefix_selection_row(self : Scheduler, row : Int) -> Unit {
  let offset = self.prefix_selection_offsets[row]
  for index in 0..<self.prefix_selection_counts[row] {
    try! self.prefix_selection_pages.clear(offset + index)
  }
  self.prefix_selection_offsets[row] = 0
  self.prefix_selection_counts[row] = 0
  self.prefix_selection_entries[row] = NO_INDEX
  self.prefix_selection_entry_generations[row] = 0
}

///|
fn Scheduler::rollback_selected_prefixes(self : Scheduler) -> Unit {
  for row in 0..<self.prefill_count {
    let slot = self.prefill_slots[row]
    let offset = self.prefix_selection_offsets[row]
    for index in 0..<self.prefix_selection_counts[row] {
      try! self.page_allocator.release_active(
        try! self.prefix_selection_pages.read(offset + index),
      )
    }
    let entry_index = self.prefix_selection_entries[row]
    if entry_index != NO_INDEX {
      try! self.prefix_index
      .unwrap()
      .release_active(
        try! @radix.PrefixEntryId::new(
          index=entry_index,
          generation=self.prefix_selection_entry_generations[row],
        ),
      )
      self.prefix_entry_indices[slot] = NO_INDEX
      self.prefix_entry_generations[slot] = 0
      self.prefix_matched_tokens[slot] = 0
      if self.recompute_required[slot] {
        self.recompute_processed[slot] = 0
      } else {
        self.prefill_cursors[slot] = 0
      }
    }
    self.clear_prefix_selection_row(row)
  }
  self.prefix_selection_page_count = 0
}

///|
fn Scheduler::commit_selected_prefixes(self : Scheduler) -> Unit {
  for row in 0..<self.prefill_count {
    if self.prefix_selection_entries[row] != NO_INDEX {
      let slot = self.prefill_slots[row]
      self.prefix_hits = Scheduler::prefix_saturating_add(self.prefix_hits, 1)
      self.prefix_tokens_reused = Scheduler::prefix_saturating_add(
        self.prefix_tokens_reused,
        self.prefix_matched_tokens[slot],
      )
    }
    self.clear_prefix_selection_row(row)
  }
  self.prefix_selection_page_count = 0
}

///|
fn Scheduler::release_prefix_entry(self : Scheduler, slot : Int) -> Unit {
  let index = self.prefix_entry_indices[slot]
  if index != NO_INDEX {
    let entry = try! @radix.PrefixEntryId::new(
      index~,
      generation=self.prefix_entry_generations[slot],
    )
    try! self.prefix_index.unwrap().release_active(entry)
    self.prefix_entry_indices[slot] = NO_INDEX
    self.prefix_entry_generations[slot] = 0
  }
  self.prefix_matched_tokens[slot] = 0
}

///|
fn Scheduler::release_waiting_prefix(self : Scheduler, slot : Int) -> Unit {
  if (try! self.request_tables.is_occupied(slot)) ||
    self.prefix_entry_indices[slot] != NO_INDEX {
    abort("scheduler waiting prefix authority invariant")
  }
  self.clear_prefix_candidate(slot)
  self.prefix_matched_tokens[slot] = 0
}
