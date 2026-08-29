///|
fn PrefixIndex::eviction_page_count(self : PrefixIndex, entry : Int) -> Int {
  let mut count = 0
  let mut node = self.entry_terminals[entry]
  while node >= 0 {
    if self.node_usages[node] == 1 {
      let mut cell = self.node_label_heads[node]
      for offset in 0..<self.node_label_lengths[node] {
        if self.token_page_cells[cell] >= 0 {
          count += 1
        }
        if offset + 1 < self.node_label_lengths[node] {
          cell = self.token_next[cell]
        }
      }
    }
    node = self.node_parents[node]
  }
  count
}

///|
fn PrefixIndex::plan_label_pages(
  self : PrefixIndex,
  node : Int,
  output : PrefixEvictionBuffer,
) -> Unit {
  let mut cell = self.node_label_heads[node]
  for offset in 0..<self.node_label_lengths[node] {
    let page_cell = self.token_page_cells[cell]
    if page_cell >= 0 {
      try! output.pages.write(
        output.page_count,
        try! self.page_values.read(page_cell),
      )
      output.page_count += 1
    }
    if offset + 1 < self.node_label_lengths[node] {
      cell = self.token_next[cell]
    }
  }
}

///|
fn PrefixIndex::free_planned_label(
  self : PrefixIndex,
  node : Int,
  output : PrefixEvictionBuffer,
  page_cursor : Int,
) -> Int {
  let mut cursor = page_cursor
  let mut cell = self.node_label_heads[node]
  for offset in 0..<self.node_label_lengths[node] {
    let next = self.token_next[cell]
    let page_cell = self.token_page_cells[cell]
    if page_cell >= 0 {
      let page = try! self.page_values.read(page_cell)
      if (try! output.page_at(cursor)) != page {
        abort("prefix eviction plan invariant")
      }
      try! self.page_values.clear(page_cell)
      self.release_page_cell(page_cell)
      self.page_count -= 1
      cursor += 1
    }
    self.token_used[cell] = false
    self.token_values[cell] = 0
    self.token_next[cell] = -1
    self.token_page_cells[cell] = -1
    self.release_token_cell(cell)
    self.token_count -= 1
    cell = next
    ignore(offset)
  }
  cursor
}

///|
fn PrefixIndex::free_node_descriptor(self : PrefixIndex, node : Int) -> Unit {
  self.detach_node(node)
  self.node_used[node] = false
  self.node_parents[node] = -1
  self.node_roots[node] = -1
  self.node_usages[node] = 0
  self.node_terminals[node] = -1
  self.node_label_heads[node] = -1
  self.node_label_tails[node] = -1
  self.node_label_lengths[node] = 0
  self.node_child_heads[node] = -1
  self.node_child_counts[node] = 0
  self.node_sibling_prev[node] = -1
  self.node_sibling_next[node] = -1
  self.node_protecting_entries[node] = -1
  self.node_count -= 1
  self.release_node_slot(node)
}

///|
fn PrefixIndex::release_merged_node_descriptor(
  self : PrefixIndex,
  node : Int,
) -> Unit {
  self.node_used[node] = false
  self.node_parents[node] = -1
  self.node_roots[node] = -1
  self.node_usages[node] = 0
  self.node_terminals[node] = -1
  self.node_label_heads[node] = -1
  self.node_label_tails[node] = -1
  self.node_label_lengths[node] = 0
  self.node_child_heads[node] = -1
  self.node_child_counts[node] = 0
  self.node_sibling_prev[node] = -1
  self.node_sibling_next[node] = -1
  self.node_protecting_entries[node] = -1
  self.node_count -= 1
  self.release_node_slot(node)
}

///|
fn PrefixIndex::merge_unary_node(self : PrefixIndex, node : Int) -> Int {
  let child = self.single_child(node)
  if child < 0 || self.node_terminals[node] >= 0 {
    abort("prefix unary merge invariant")
  }
  let parent = self.node_parents[node]
  let root = self.node_roots[node]
  let label_head = self.node_label_heads[node]
  let label_tail = self.node_label_tails[node]
  let label_length = self.node_label_lengths[node]
  self.detach_node(child)
  self.detach_node(node)
  self.token_next[label_tail] = self.node_label_heads[child]
  self.node_label_heads[child] = label_head
  self.node_label_lengths[child] += label_length
  self.node_parents[child] = parent
  self.node_roots[child] = root
  self.node_usages[child] = self.node_usages[node]
  self.attach_node(child)
  self.release_merged_node_descriptor(node)
  child
}

///|
fn PrefixIndex::compact_changed_path(self : PrefixIndex, start : Int) -> Unit {
  let mut node = start
  while node >= 0 {
    self.compaction_node_visits += 1
    let parent = self.node_parents[node]
    if self.node_terminals[node] < 0 && self.node_child_counts[node] == 1 {
      let child = self.merge_unary_node(node)
      self.refresh_node_protector(child)
    } else {
      self.refresh_node_protector(node)
    }
    node = parent
  }
}

///|
fn PrefixIndex::require_eviction_plan(
  self : PrefixIndex,
  plan : PrefixEvictionPlan,
) -> Unit raise PrefixIndexError {
  if !physical_equal(plan.owner, self) ||
    !self.eviction_plan_active ||
    plan.generation != self.eviction_plan_generation {
    raise StaleEvictionPlan
  }
}

///|
fn PrefixIndex::clear_eviction_plan(self : PrefixIndex) -> Unit {
  for index in 0..<self.eviction_plan_page_count {
    try! self.eviction_page_scratch.clear(index)
  }
  self.eviction_plan_active = false
  self.eviction_plan_entry_index = -1
  self.eviction_plan_entry_generation = 0
  self.eviction_plan_page_count = 0
}

///|
fn PrefixIndex::eviction_plan_matches(
  self : PrefixIndex,
  output : PrefixEvictionBuffer,
) -> Bool {
  let selected = self.eviction_plan_entry_index
  if selected < 0 ||
    selected >= self.entry_states.length() ||
    self.entry_states[selected] != ENTRY_USED ||
    self.entry_generations[selected] != self.eviction_plan_entry_generation ||
    self.entry_active_references[selected] != 0 ||
    self.eviction_candidate() != selected ||
    self.eviction_page_count(selected) != self.eviction_plan_page_count ||
    output.page_count() != self.eviction_plan_page_count ||
    output.entry_index != selected ||
    output.entry_generation != self.eviction_plan_entry_generation {
    return false
  }
  for index in 0..<output.page_count() {
    if (try! output.page_at(index)) !=
      (try! self.eviction_page_scratch.read(index)) {
      return false
    }
  }
  true
}

///|
pub fn PrefixEvictionStart::is_planned(self : PrefixEvictionStart) -> Bool {
  self.state == 0
}

///|
pub fn PrefixEvictionStart::is_absent(self : PrefixEvictionStart) -> Bool {
  self.state == 1
}

///|
pub fn PrefixEvictionStart::is_rejected(self : PrefixEvictionStart) -> Bool {
  self.state == 2
}

///|
pub fn PrefixEvictionStart::plan(
  self : PrefixEvictionStart,
) -> PrefixEvictionPlan raise PrefixIndexError {
  if self.state != 0 {
    raise StaleEvictionPlan
  }
  { owner: self.owner, generation: self.generation }
}

///|
/// Selects one deterministic zero-reference victim without constructing an
/// error for an empty cache or a terminal plan-generation owner.
pub fn PrefixIndex::try_plan_eviction(
  self : PrefixIndex,
  output : PrefixEvictionBuffer,
) -> PrefixEvictionStart raise PrefixIndexError {
  if self.publish_plan_active || self.eviction_plan_active {
    raise EvictionPlanBusy
  }
  if self.eviction_plan_generation == 0xffffffffffffffffUL {
    return { state: 2, owner: self, generation: self.eviction_plan_generation }
  }
  let selected = self.eviction_candidate()
  if selected < 0 {
    return { state: 1, owner: self, generation: self.eviction_plan_generation }
  }
  let removed_pages = self.eviction_page_count(selected)
  if removed_pages > output.capacity() {
    raise ResultBufferTooSmall
  }
  output.clear()
  output.entry_index = selected
  output.entry_generation = self.entry_generations[selected]
  let mut node = self.entry_terminals[selected]
  while node >= 0 {
    if self.node_usages[node] == 1 {
      self.plan_label_pages(node, output)
    }
    node = self.node_parents[node]
  }
  for index in 0..<output.page_count() {
    try! self.eviction_page_scratch.write(index, try! output.page_at(index))
  }
  self.eviction_plan_generation += 1UL
  self.eviction_plan_active = true
  self.eviction_plan_entry_index = selected
  self.eviction_plan_entry_generation = self.entry_generations[selected]
  self.eviction_plan_page_count = output.page_count()
  { state: 0, owner: self, generation: self.eviction_plan_generation }
}

///|
pub fn PrefixIndex::commit_eviction(
  self : PrefixIndex,
  plan : PrefixEvictionPlan,
  output : PrefixEvictionBuffer,
) -> PrefixEntryId raise PrefixIndexError {
  self.require_eviction_plan(plan)
  if !self.eviction_plan_matches(output) {
    raise EvictionPlanInvalidated
  }
  let selected = self.eviction_plan_entry_index
  let evicted = PrefixEntryId(selected, self.eviction_plan_entry_generation)
  let root = self.entry_roots[selected]
  let terminal = self.entry_terminals[selected]
  self.eviction_heap_remove(selected)
  self.node_terminals[terminal] = -1
  let mut node = terminal
  let mut page_cursor = 0
  let mut surviving_parent = -1
  while node >= 0 {
    let parent = self.node_parents[node]
    self.compaction_node_visits += 1
    self.node_usages[node] -= 1
    if self.node_usages[node] == 0 {
      page_cursor = self.free_planned_label(node, output, page_cursor)
      self.free_node_descriptor(node)
    } else if surviving_parent < 0 {
      surviving_parent = node
    }
    node = parent
  }
  self.root_entry_counts[root] -= 1
  if self.root_entry_counts[root] == 0 {
    self.root_index_remove(root)
    let scope_length = self.root_scope_lengths[root]
    for index in 0..<scope_length {
      self.root_scope_bytes[root * self.limits.max_scope_bytes() + index] = b'\x00'
    }
    self.root_used[root] = false
    self.root_models[root] = None
    self.root_tokenizers[root] = None
    self.root_layouts[root] = None
    self.root_scope_lengths[root] = 0
    self.root_child_heads[root] = -1
    self.root_child_counts[root] = 0
    self.release_root_slot(root)
  } else {
    self.compact_changed_path(surviving_parent)
  }
  self.entry_roots[selected] = -1
  self.entry_terminals[selected] = -1
  self.entry_token_counts[selected] = 0
  self.entry_page_counts[selected] = 0
  self.entry_priorities[selected] = 0
  self.entry_active_references[selected] = 0
  self.entry_recencies[selected] = 0UL
  if self.entry_generations[selected] == self.limits.max_generation() {
    self.entry_states[selected] = ENTRY_RETIRED
    self.retired_entry_count += 1
  } else {
    self.entry_generations[selected] += 1
    self.entry_states[selected] = ENTRY_FREE
    self.release_entry_slot(selected)
  }
  self.entry_count -= 1
  self.clear_eviction_plan()
  evicted
}

///|
pub fn PrefixIndex::abort_eviction(
  self : PrefixIndex,
  plan : PrefixEvictionPlan,
) -> Unit raise PrefixIndexError {
  self.require_eviction_plan(plan)
  self.clear_eviction_plan()
}
