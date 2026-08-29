///|
const ENTRY_FREE : Int = 0

///|
const ENTRY_USED : Int = 1

///|
const ENTRY_RETIRED : Int = 2

///|
pub fn PrefixIndex::new(
  limits : PrefixIndexLimits,
) -> PrefixIndex raise PrefixIndexError {
  ignore(PrefixIndex::required_int_cells(limits))
  ignore(PrefixIndex::required_bool_cells(limits))
  ignore(PrefixIndex::required_byte_cells(limits))
  ignore(PrefixIndex::required_reference_cells(limits))
  let entries = limits.entry_capacity()
  let nodes = limits.node_capacity()
  let tokens = limits.token_capacity()
  let pages = limits.page_capacity()
  let root_free_next = FixedArray::make(entries, -1)
  let root_index_free_next = FixedArray::make(entries, -1)
  let entry_free_next = FixedArray::make(entries, -1)
  for index in 0..<entries {
    let next = if index + 1 < entries { index + 1 } else { -1 }
    root_free_next[index] = next
    root_index_free_next[index] = next
    entry_free_next[index] = next
  }
  let node_free_next = FixedArray::make(nodes, -1)
  let child_index_free_next = FixedArray::make(nodes, -1)
  for index in 0..<nodes {
    let next = if index + 1 < nodes { index + 1 } else { -1 }
    node_free_next[index] = next
    child_index_free_next[index] = next
  }
  let token_free_next = FixedArray::make(tokens, -1)
  for index in 0..<tokens {
    token_free_next[index] = if index + 1 < tokens { index + 1 } else { -1 }
  }
  let page_free_next = FixedArray::make(pages, -1)
  for index in 0..<pages {
    page_free_next[index] = if index + 1 < pages { index + 1 } else { -1 }
  }
  {
    limits,
    root_used: FixedArray::make(entries, false),
    root_models: FixedArray::make(entries, None),
    root_tokenizers: FixedArray::make(entries, None),
    root_layouts: FixedArray::make(entries, None),
    root_scope_lengths: FixedArray::make(entries, 0),
    root_scope_bytes: FixedArray::make(
      entries * limits.max_scope_bytes(),
      b'\x00',
    ),
    root_scope_scratch: FixedArray::make(limits.max_scope_bytes(), b'\x00'),
    root_entry_counts: FixedArray::make(entries, 0),
    root_child_heads: FixedArray::make(entries, -1),
    root_child_counts: FixedArray::make(entries, 0),
    root_free_next,
    root_free_head: 0,
    root_index_payloads: FixedArray::make(entries, -1),
    root_index_left: FixedArray::make(entries, -1),
    root_index_right: FixedArray::make(entries, -1),
    root_index_parents: FixedArray::make(entries, -1),
    root_index_heights: FixedArray::make(entries, 0),
    root_index_slots: FixedArray::make(entries, -1),
    root_index_free_next,
    root_index_root: -1,
    root_index_free_head: 0,
    node_used: FixedArray::make(nodes, false),
    node_parents: FixedArray::make(nodes, -1),
    node_roots: FixedArray::make(nodes, -1),
    node_usages: FixedArray::make(nodes, 0),
    node_terminals: FixedArray::make(nodes, -1),
    node_label_heads: FixedArray::make(nodes, -1),
    node_label_tails: FixedArray::make(nodes, -1),
    node_label_lengths: FixedArray::make(nodes, 0),
    node_child_heads: FixedArray::make(nodes, -1),
    node_child_counts: FixedArray::make(nodes, 0),
    node_sibling_prev: FixedArray::make(nodes, -1),
    node_sibling_next: FixedArray::make(nodes, -1),
    node_protecting_entries: FixedArray::make(nodes, -1),
    node_free_next,
    node_free_head: 0,
    child_index_payloads: FixedArray::make(nodes, -1),
    child_index_left: FixedArray::make(nodes, -1),
    child_index_right: FixedArray::make(nodes, -1),
    child_index_parents: FixedArray::make(nodes, -1),
    child_index_heights: FixedArray::make(nodes, 0),
    child_index_slots: FixedArray::make(nodes, -1),
    child_index_free_next,
    child_index_root: -1,
    child_index_free_head: 0,
    node_path_scratch: FixedArray::make(nodes, -1),
    lookup_page_scratch: FixedArray::make(pages, -1),
    publish_token_scratch: FixedArray::make(tokens, 0),
    publish_page_scratch: @page_allocator.PageIdStorage::new(pages) catch {
      _ => raise InvalidLimit(PageCapacity)
    },
    eviction_page_scratch: @page_allocator.PageIdStorage::new(pages) catch {
      _ => raise InvalidLimit(PageCapacity)
    },
    token_used: FixedArray::make(tokens, false),
    token_values: FixedArray::make(tokens, 0),
    token_next: FixedArray::make(tokens, -1),
    token_page_cells: FixedArray::make(tokens, -1),
    token_free_next,
    token_free_head: 0,
    entry_states: FixedArray::make(entries, ENTRY_FREE),
    entry_generations: FixedArray::make(entries, 1),
    entry_roots: FixedArray::make(entries, -1),
    entry_terminals: FixedArray::make(entries, -1),
    entry_token_counts: FixedArray::make(entries, 0),
    entry_page_counts: FixedArray::make(entries, 0),
    entry_priorities: FixedArray::make(entries, 0),
    entry_active_references: FixedArray::make(entries, 0),
    entry_recencies: FixedArray::make(entries, 0UL),
    entry_free_next,
    entry_free_head: 0,
    eviction_heap: FixedArray::make(entries, -1),
    eviction_heap_positions: FixedArray::make(entries, -1),
    eviction_heap_count: 0,
    recency_entry_scratch: FixedArray::make(entries, -1),
    page_values: @page_allocator.PageIdStorage::new(pages) catch {
      _ => raise InvalidLimit(PageCapacity)
    },
    duplicate_page_scratch: @page_allocator.PageIdStorage::new(pages) catch {
      _ => raise InvalidLimit(PageCapacity)
    },
    page_free_next,
    page_free_head: 0,
    entry_count: 0,
    retired_entry_count: 0,
    node_count: 0,
    token_count: 0,
    page_count: 0,
    recency_clock: 0UL,
    publish_plan_generation: 0UL,
    publish_plan_active: false,
    publish_plan_root: -1,
    publish_plan_token_count: 0,
    publish_plan_page_count: 0,
    publish_plan_priority: 0,
    eviction_plan_generation: 0UL,
    eviction_plan_active: false,
    eviction_plan_entry_index: -1,
    eviction_plan_entry_generation: 0,
    eviction_plan_page_count: 0,
    child_lookup_comparisons: 0,
    root_lookup_comparisons: 0,
    duplicate_page_comparisons: 0,
    duplicate_page_swaps: 0,
    recency_normalization_comparisons: 0,
    recency_normalization_swaps: 0,
    free_stack_touches: 0,
    eviction_heap_touches: 0,
    compaction_node_visits: 0,
    compaction_incident_edges: 0,
  }
}

///|
pub fn PrefixIndex::limits(self : PrefixIndex) -> PrefixIndexLimits {
  self.limits
}

///|
pub fn PrefixIndex::entry_count(self : PrefixIndex) -> Int {
  self.entry_count
}

///|
pub fn PrefixIndex::retired_entry_count(self : PrefixIndex) -> Int {
  self.retired_entry_count
}

///|
pub fn PrefixIndex::node_count(self : PrefixIndex) -> Int {
  self.node_count
}

///|
pub fn PrefixIndex::token_cell_count(self : PrefixIndex) -> Int {
  self.token_count
}

///|
pub fn PrefixIndex::page_count(self : PrefixIndex) -> Int {
  self.page_count
}

///|
fn PrefixIndex::root_scope_byte(
  self : PrefixIndex,
  root : Int,
  index : Int,
) -> Byte {
  self.root_scope_bytes[root * self.limits.max_scope_bytes() + index]
}

///|
fn PrefixIndex::request_root_matches(
  self : PrefixIndex,
  root : Int,
  key : PrefixRequestKey,
) -> Bool raise PrefixIndexError {
  if !key.is_live() {
    raise StaleKey
  }
  if !self.root_used[root] ||
    self.root_models[root] != Some(key.model_identity) ||
    self.root_tokenizers[root] != Some(key.tokenizer) ||
    self.root_layouts[root] != Some(key.layout) {
    return false
  }
  let length = key.semantic.scope_length() catch { _ => raise StaleKey }
  if length != self.root_scope_lengths[root] {
    return false
  }
  for index in 0..<length {
    let byte = key.semantic.scope_byte_at(index) catch { _ => raise StaleKey }
    if self.root_scope_byte(root, index) != byte {
      return false
    }
  }
  true
}

///|
fn PrefixIndex::free_root(self : PrefixIndex) -> Int {
  self.root_free_head
}

///|
fn PrefixIndex::node_first_token(self : PrefixIndex, node : Int) -> Int {
  self.token_values[self.node_label_heads[node]]
}

///|
fn PrefixIndex::checked_entry(
  self : PrefixIndex,
  entry : PrefixEntryId,
) -> Int raise PrefixIndexError {
  let index = entry.index()
  if index < 0 ||
    index >= self.entry_states.length() ||
    self.entry_states[index] != ENTRY_USED ||
    self.entry_generations[index] != entry.generation() {
    raise StaleEntry(entry)
  }
  index
}

///|
fn PrefixIndex::next_recency(self : PrefixIndex) -> UInt64 {
  if self.recency_clock == 0xffffffffffffffffUL {
    self.normalize_recencies()
  }
  self.recency_clock += 1UL
  self.recency_clock
}

///|
