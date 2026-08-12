///|
pub fn PrefixIndex::evict_oldest_zero_reference(
  self : PrefixIndex,
  output : PrefixEvictionBuffer,
) -> PrefixEntryId raise PrefixIndexError {
  let mut selected = -1
  let mut oldest = 0UL
  for index in 0..<self.entry_states.length() {
    if self.entry_states[index] == ENTRY_USED &&
      self.entry_active_references[index] == 0 &&
      (selected < 0 || self.entry_recencies[index] < oldest) {
      selected = index
      oldest = self.entry_recencies[index]
    }
  }
  if selected < 0 {
    raise NoEvictableEntry
  }
  let count = self.entry_page_counts[selected]
  if count > output.capacity() {
    raise ResultBufferTooSmall
  }
  let mut checked_cell = self.entry_page_heads[selected]
  for _ in 0..<count {
    if checked_cell < 0 ||
      checked_cell >= self.page_used.length() ||
      !self.page_used[checked_cell] ||
      self.page_values[checked_cell] is None {
      raise InvariantViolation(PageChain)
    }
    checked_cell = self.page_next[checked_cell]
  }
  if checked_cell != -1 {
    raise InvariantViolation(PageChain)
  }
  let evicted = PrefixEntryId(selected, self.entry_generations[selected])
  output.clear()
  let mut cell = self.entry_page_heads[selected]
  for page_index in 0..<count {
    guard self.page_values[cell] is Some(page) else {
      abort("validated prefix page disappeared")
    }
    output.pages[page_index] = Some(page)
    cell = self.page_next[cell]
  }
  output.page_count = count
  output.entry = Some(evicted)
  cell = self.entry_page_heads[selected]
  for _ in 0..<count {
    let next = self.page_next[cell]
    self.page_used[cell] = false
    self.page_values[cell] = None
    self.page_next[cell] = -1
    self.page_count = self.page_count - 1
    cell = next
  }
  let terminal = self.entry_terminals[selected]
  self.node_terminals[terminal] = -1
  let mut node = terminal
  for _ in 0..<self.entry_token_counts[selected] {
    let parent = self.node_parents[node]
    self.node_usages[node] = self.node_usages[node] - 1
    if self.node_usages[node] == 0 {
      self.node_used[node] = false
      self.node_tokens[node] = 0
      self.node_parents[node] = -1
      self.node_roots[node] = -1
      self.node_terminals[node] = -1
      self.node_count = self.node_count - 1
    }
    node = parent
  }
  let root = self.entry_roots[selected]
  self.root_entry_counts[root] = self.root_entry_counts[root] - 1
  if self.root_entry_counts[root] == 0 {
    self.roots[root] = None
  }
  self.entry_roots[selected] = -1
  self.entry_terminals[selected] = -1
  self.entry_token_counts[selected] = 0
  self.entry_page_heads[selected] = -1
  self.entry_page_counts[selected] = 0
  self.entry_active_references[selected] = 0
  self.entry_recencies[selected] = 0UL
  if self.entry_generations[selected] == self.limits.max_generation() {
    self.entry_states[selected] = ENTRY_RETIRED
    self.retired_entry_count = self.retired_entry_count + 1
  } else {
    self.entry_generations[selected] = self.entry_generations[selected] + 1
    self.entry_states[selected] = ENTRY_FREE
  }
  self.entry_count = self.entry_count - 1
  evicted
}
