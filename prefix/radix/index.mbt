///|
const ENTRY_FREE : Int = 0

///|
const ENTRY_USED : Int = 1

///|
const ENTRY_RETIRED : Int = 2

///|
pub fn PrefixIndex::new(limits : PrefixIndexLimits) -> PrefixIndex {
  let entries = limits.entry_capacity()
  let nodes = limits.node_capacity()
  let pages = limits.page_capacity()
  {
    limits,
    roots: FixedArray::make(entries, None),
    root_entry_counts: FixedArray::make(entries, 0),
    node_used: FixedArray::make(nodes, false),
    node_tokens: FixedArray::make(nodes, 0),
    node_parents: FixedArray::make(nodes, -1),
    node_roots: FixedArray::make(nodes, -1),
    node_usages: FixedArray::make(nodes, 0),
    node_terminals: FixedArray::make(nodes, -1),
    entry_states: FixedArray::make(entries, ENTRY_FREE),
    entry_generations: FixedArray::make(entries, 1),
    entry_roots: FixedArray::make(entries, -1),
    entry_terminals: FixedArray::make(entries, -1),
    entry_token_counts: FixedArray::make(entries, 0),
    entry_page_heads: FixedArray::make(entries, -1),
    entry_page_counts: FixedArray::make(entries, 0),
    entry_active_references: FixedArray::make(entries, 0),
    entry_recencies: FixedArray::make(entries, 0UL),
    recency_scratch: FixedArray::make(entries, 0UL),
    page_used: FixedArray::make(pages, false),
    page_values: FixedArray::make(pages, None),
    page_next: FixedArray::make(pages, -1),
    entry_count: 0,
    retired_entry_count: 0,
    node_count: 0,
    page_count: 0,
    recency_clock: 0UL,
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
pub fn PrefixIndex::page_count(self : PrefixIndex) -> Int {
  self.page_count
}

///|
fn identities_equal(left : PrefixIdentity, right : PrefixIdentity) -> Bool {
  left.model_identity() == right.model_identity() &&
  left.tokenizer() == right.tokenizer() &&
  left.scope().as_string() == right.scope().as_string() &&
  left.layout() == right.layout()
}

///|
fn PrefixIndex::find_root(self : PrefixIndex, identity : PrefixIdentity) -> Int {
  for index in 0..<self.roots.length() {
    match self.roots[index] {
      Some(candidate) if identities_equal(candidate, identity) => return index
      _ => ()
    }
  }
  -1
}

///|
fn PrefixIndex::free_root(self : PrefixIndex) -> Int {
  for index in 0..<self.roots.length() {
    if self.roots[index] is None {
      return index
    }
  }
  -1
}

///|
fn PrefixIndex::find_child(
  self : PrefixIndex,
  root : Int,
  parent : Int,
  token : Int,
) -> Int {
  for index in 0..<self.node_used.length() {
    if self.node_used[index] &&
      self.node_roots[index] == root &&
      self.node_parents[index] == parent &&
      self.node_tokens[index] == token {
      return index
    }
  }
  -1
}

///|
fn PrefixIndex::free_entry(self : PrefixIndex) -> Int {
  for index in 0..<self.entry_states.length() {
    if self.entry_states[index] == ENTRY_FREE {
      return index
    }
  }
  -1
}

///|
fn PrefixIndex::free_node_count(self : PrefixIndex) -> Int {
  self.node_used.length() - self.node_count
}

///|
fn PrefixIndex::free_page_cell_count(self : PrefixIndex) -> Int {
  self.page_used.length() - self.page_count
}

///|
fn PrefixIndex::normalize_recencies(self : PrefixIndex) -> Unit {
  for index in 0..<self.entry_states.length() {
    self.recency_scratch[index] = self.entry_recencies[index]
  }
  for index in 0..<self.entry_states.length() {
    if self.entry_states[index] == ENTRY_USED {
      let mut rank = 1
      for candidate in 0..<self.entry_states.length() {
        if self.entry_states[candidate] == ENTRY_USED &&
          (
            self.recency_scratch[candidate] < self.recency_scratch[index] ||
            (
              self.recency_scratch[candidate] == self.recency_scratch[index] &&
              candidate < index
            )
          ) {
          rank = rank + 1
        }
      }
      self.entry_recencies[index] = Int64::from_int(rank).reinterpret_as_uint64()
    }
  }
  self.recency_clock = Int64::from_int(self.entry_count).reinterpret_as_uint64()
}

///|
fn PrefixIndex::next_recency(self : PrefixIndex) -> UInt64 {
  if self.recency_clock == 0xffffffffffffffffUL {
    self.normalize_recencies()
  }
  self.recency_clock = self.recency_clock + 1UL
  self.recency_clock
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
pub fn PrefixIndex::insert(
  self : PrefixIndex,
  identity : PrefixIdentity,
  tokens : ArrayView[Int],
  pages : ArrayView[@page_allocator.PageId],
) -> PrefixEntryId raise PrefixIndexError {
  let token_count = tokens.length()
  let page_count = pages.length()
  let tokens_per_page = identity.layout().tokens_per_page()
  if token_count <= 0 || token_count > self.limits.max_tokens_per_entry() {
    raise InvalidTokenCount
  }
  for token_index, token in tokens {
    if token < 0 {
      raise InvalidToken(token_index)
    }
  }
  if page_count <= 0 ||
    page_count > self.limits.page_capacity() ||
    page_count > self.limits.max_tokens_per_entry() / tokens_per_page ||
    token_count != page_count * tokens_per_page {
    raise InvalidPageCount
  }
  let existing_root = self.find_root(identity)
  let root = if existing_root >= 0 { existing_root } else { self.free_root() }
  let entry_index = self.free_entry()
  if root < 0 || entry_index < 0 {
    raise Exhausted
  }
  let mut parent = -1
  let mut existing_tokens = 0
  if existing_root >= 0 {
    for token in tokens {
      let child = self.find_child(root, parent, token)
      if child < 0 {
        break
      }
      parent = child
      existing_tokens = existing_tokens + 1
    }
  }
  if existing_tokens == token_count {
    let terminal = self.node_terminals[parent]
    if terminal >= 0 {
      raise DuplicatePrefix(
        PrefixEntryId(terminal, self.entry_generations[terminal]),
      )
    }
  }
  let missing_nodes = token_count - existing_tokens
  if missing_nodes > self.free_node_count() ||
    page_count > self.free_page_cell_count() {
    raise Exhausted
  }
  let recency = self.next_recency()
  if existing_root < 0 {
    self.roots[root] = Some(identity)
  }
  parent = -1
  for token_index, token in tokens {
    let mut node = self.find_child(root, parent, token)
    if node < 0 {
      for candidate in 0..<self.node_used.length() {
        if !self.node_used[candidate] {
          node = candidate
          break
        }
      }
      self.node_used[node] = true
      self.node_tokens[node] = token
      self.node_parents[node] = parent
      self.node_roots[node] = root
      self.node_usages[node] = 0
      self.node_terminals[node] = -1
      self.node_count = self.node_count + 1
    }
    self.node_usages[node] = self.node_usages[node] + 1
    parent = node
    if token_index + 1 == token_count {
      self.node_terminals[node] = entry_index
    }
  }
  let mut page_head = -1
  let mut page_tail = -1
  for page in pages {
    let mut cell = -1
    for candidate in 0..<self.page_used.length() {
      if !self.page_used[candidate] {
        cell = candidate
        break
      }
    }
    self.page_used[cell] = true
    self.page_values[cell] = Some(page)
    self.page_next[cell] = -1
    if page_head < 0 {
      page_head = cell
    } else {
      self.page_next[page_tail] = cell
    }
    page_tail = cell
    self.page_count = self.page_count + 1
  }
  self.entry_states[entry_index] = ENTRY_USED
  self.entry_roots[entry_index] = root
  self.entry_terminals[entry_index] = parent
  self.entry_token_counts[entry_index] = token_count
  self.entry_page_heads[entry_index] = page_head
  self.entry_page_counts[entry_index] = page_count
  self.entry_active_references[entry_index] = 0
  self.entry_recencies[entry_index] = recency
  self.root_entry_counts[root] = self.root_entry_counts[root] + 1
  self.entry_count = self.entry_count + 1
  PrefixEntryId(entry_index, self.entry_generations[entry_index])
}

///|
pub fn PrefixIndex::lookup_longest_full_pages(
  self : PrefixIndex,
  identity : PrefixIdentity,
  tokens : ArrayView[Int],
  output : PrefixLookupBuffer,
) -> Unit raise PrefixIndexError {
  if tokens.length() > self.limits.max_tokens_per_entry() {
    raise InvalidTokenCount
  }
  for token_index, token in tokens {
    if token < 0 {
      raise InvalidToken(token_index)
    }
  }
  let root = self.find_root(identity)
  if root < 0 || tokens.length() == 0 {
    output.clear()
    return
  }
  let page_size = identity.layout().tokens_per_page()
  let usable_tokens = tokens.length() / page_size * page_size
  let mut parent = -1
  let mut best_entry = -1
  let mut matched = 0
  for index in 0..<usable_tokens {
    let child = self.find_child(root, parent, tokens[index])
    if child < 0 {
      break
    }
    parent = child
    if (index + 1) % page_size == 0 && self.node_terminals[child] >= 0 {
      best_entry = self.node_terminals[child]
      matched = index + 1
    }
  }
  if best_entry < 0 {
    output.clear()
    return
  }
  let count = self.entry_page_counts[best_entry]
  if count > output.capacity() {
    raise ResultBufferTooSmall
  }
  let mut checked_cell = self.entry_page_heads[best_entry]
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
  let recency = self.next_recency()
  output.clear()
  let mut cell = self.entry_page_heads[best_entry]
  for page_index in 0..<count {
    guard self.page_values[cell] is Some(page) else {
      abort("validated prefix page disappeared")
    }
    output.pages[page_index] = Some(page)
    cell = self.page_next[cell]
  }
  output.page_count = count
  output.matched_tokens = matched
  output.entry = Some(
    PrefixEntryId(best_entry, self.entry_generations[best_entry]),
  )
  self.entry_recencies[best_entry] = recency
}

///|
pub fn PrefixIndex::retain_active(
  self : PrefixIndex,
  entry : PrefixEntryId,
) -> Unit raise PrefixIndexError {
  let index = self.checked_entry(entry)
  if self.entry_active_references[index] >=
    self.limits.max_active_references_per_entry() {
    raise ReferenceOverflow(entry)
  }
  self.entry_active_references[index] = self.entry_active_references[index] + 1
}

///|
pub fn PrefixIndex::release_active(
  self : PrefixIndex,
  entry : PrefixEntryId,
) -> Unit raise PrefixIndexError {
  let index = self.checked_entry(entry)
  if self.entry_active_references[index] == 0 {
    raise ReferenceUnderflow(entry)
  }
  self.entry_active_references[index] = self.entry_active_references[index] - 1
}

///|
pub fn PrefixIndex::active_reference_count(
  self : PrefixIndex,
  entry : PrefixEntryId,
) -> Int raise PrefixIndexError {
  self.entry_active_references[self.checked_entry(entry)]
}
