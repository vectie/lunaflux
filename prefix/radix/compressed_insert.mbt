///|
fn PrefixIndex::free_node_count(self : PrefixIndex) -> Int {
  self.node_used.length() - self.node_count
}

///|
fn PrefixIndex::free_token_count(self : PrefixIndex) -> Int {
  self.token_used.length() - self.token_count
}

///|
fn PrefixIndex::free_page_count(self : PrefixIndex) -> Int {
  self.limits.page_capacity() - self.page_count
}

///|
fn PrefixIndex::allocate_label(
  self : PrefixIndex,
  tokens : PrefixTokenBuffer,
  start : Int,
  length : Int,
) -> Int {
  let mut head = -1
  let mut tail = -1
  for offset in 0..<length {
    let cell = self.take_free_token()
    if cell < 0 {
      abort("prefix token preflight invariant")
    }
    self.token_used[cell] = true
    self.token_values[cell] = tokens.token_at(start + offset)
    self.token_next[cell] = -1
    self.token_page_cells[cell] = -1
    if head < 0 {
      head = cell
    } else {
      self.token_next[tail] = cell
    }
    tail = cell
    self.token_count += 1
  }
  head
}

///|
fn PrefixIndex::allocate_node(
  self : PrefixIndex,
  root : Int,
  parent : Int,
  label_head : Int,
  label_length : Int,
) -> Int {
  let node = self.take_free_node()
  if node < 0 {
    abort("prefix node preflight invariant")
  }
  self.node_used[node] = true
  self.node_roots[node] = root
  self.node_parents[node] = parent
  self.node_usages[node] = 0
  self.node_terminals[node] = -1
  self.node_label_heads[node] = label_head
  let mut tail = label_head
  for _ in 1..<label_length {
    tail = self.token_next[tail]
  }
  self.node_label_tails[node] = tail
  self.node_label_lengths[node] = label_length
  self.node_child_heads[node] = -1
  self.node_child_counts[node] = 0
  self.node_sibling_prev[node] = -1
  self.node_sibling_next[node] = -1
  self.node_protecting_entries[node] = -1
  self.node_count += 1
  self.attach_node(node)
  node
}

///|
/// Scalar insertion-capacity preflight retained entirely in registers.
#valtype
priv struct PrefixInsertionNeed {
  nodes : Int
  tokens : Int
  pages : Int
  duplicate : Bool
}

///|
/// Scalar existing-root publication preflight retained in registers.
#valtype
priv struct PrefixExistingPublishPreflight {
  state : Int
  duplicate_entry : Int
}

///|
fn PrefixIndex::validate_publish_shape(
  self : PrefixIndex,
  tokens : PrefixTokenBuffer,
  pages : PrefixPublishBuffer,
  page_size : Int,
  priority : Int,
) -> Unit raise PrefixIndexError {
  if tokens.length() <= 0 ||
    tokens.length() > self.limits.max_tokens_per_entry() {
    raise InvalidTokenCount
  }
  if priority < 0 {
    raise InvalidLimit(Priority)
  }
  if pages.length() <= 0 ||
    pages.length() > self.limits.page_capacity() ||
    tokens.length() % page_size != 0 ||
    tokens.length() / page_size != pages.length() {
    raise InvalidPageCount
  }
  if self.publish_pages_have_duplicate(pages) {
    raise InvalidPageCount
  }
}

///|
fn PrefixIndex::insertion_need(
  self : PrefixIndex,
  root : Int,
  tokens : PrefixTokenBuffer,
  page_size : Int,
) -> PrefixInsertionNeed {
  let mut parent = -1
  let mut position = 0
  let mut missing_pages = 0
  while position < tokens.length() {
    let child = self.find_child(root, parent, tokens.token_at(position))
    if child < 0 {
      let remaining = tokens.length() - position
      missing_pages += tokens.length() / page_size - position / page_size
      return {
        nodes: 1,
        tokens: remaining,
        pages: missing_pages,
        duplicate: false,
      }
    }
    let mut cell = self.node_label_heads[child]
    let mut matched = 0
    while matched < self.node_label_lengths[child] &&
          position + matched < tokens.length() &&
          self.token_values[cell] == tokens.token_at(position + matched) {
      if (position + matched + 1) % page_size == 0 &&
        self.token_page_cells[cell] < 0 {
        missing_pages += 1
      }
      matched += 1
      if matched < self.node_label_lengths[child] {
        cell = self.token_next[cell]
      }
    }
    if matched < self.node_label_lengths[child] {
      let consumed = position + matched
      let remaining = tokens.length() - consumed
      let nodes = if remaining == 0 { 1 } else { 2 }
      missing_pages += tokens.length() / page_size - consumed / page_size
      return {
        nodes,
        tokens: remaining,
        pages: missing_pages,
        duplicate: false,
      }
    }
    position += matched
    parent = child
  }
  {
    nodes: 0,
    tokens: 0,
    pages: missing_pages,
    duplicate: self.node_terminals[parent] >= 0,
  }
}

///|
fn PrefixIndex::mark_expected_adoptions(
  self : PrefixIndex,
  root : Int,
  tokens : PrefixTokenBuffer,
  page_size : Int,
  expected : FixedArray[Bool],
  page_count : Int,
) -> Unit {
  for index in 0..<page_count {
    expected[index] = false
  }
  if root < 0 {
    for index in 0..<page_count {
      expected[index] = true
    }
    return
  }
  let mut parent = -1
  let mut position = 0
  let mut diverged = false
  while position < tokens.length() && !diverged {
    let child = self.find_child(root, parent, tokens.token_at(position))
    if child < 0 {
      diverged = true
    } else {
      let mut cell = self.node_label_heads[child]
      let mut offset = 0
      while offset < self.node_label_lengths[child] &&
            position < tokens.length() {
        if self.token_values[cell] != tokens.token_at(position) {
          diverged = true
          break
        }
        position += 1
        if position % page_size == 0 && self.token_page_cells[cell] < 0 {
          expected[position / page_size - 1] = true
        }
        offset += 1
        if offset < self.node_label_lengths[child] {
          cell = self.token_next[cell]
        }
      }
      if !diverged && offset == self.node_label_lengths[child] {
        parent = child
      }
    }
  }
  if diverged {
    let first = position / page_size
    for index in first..<page_count {
      expected[index] = true
    }
  }
}

///|
/// Returns ready=0, duplicate=1, or capacity-exhausted=2 without raising.
fn PrefixIndex::existing_publish_preflight(
  self : PrefixIndex,
  root : Int,
  tokens : PrefixTokenBuffer,
  page_size : Int,
) -> PrefixExistingPublishPreflight {
  let need = self.insertion_need(root, tokens, page_size)
  if need.duplicate {
    let mut parent = -1
    let mut position = 0
    while position < tokens.length() {
      parent = self.find_child(root, parent, tokens.token_at(position))
      position += self.node_label_lengths[parent]
    }
    return { state: 1, duplicate_entry: self.node_terminals[parent] }
  }
  if self.free_entry() < 0 ||
    need.nodes > self.free_node_count() ||
    need.tokens > self.free_token_count() ||
    need.pages > self.free_page_count() {
    return { state: 2, duplicate_entry: -1 }
  }
  { state: 0, duplicate_entry: -1 }
}

///|
fn PrefixIndex::split_node(self : PrefixIndex, node : Int, common : Int) -> Int {
  let old_parent = self.node_parents[node]
  let old_head = self.node_label_heads[node]
  self.detach_node(node)
  let mut tail = old_head
  for _ in 1..<common {
    tail = self.token_next[tail]
  }
  let remainder = self.token_next[tail]
  self.token_next[tail] = -1
  let branch = self.allocate_node(
    self.node_roots[node],
    old_parent,
    old_head,
    common,
  )
  self.node_usages[branch] = self.node_usages[node]
  self.node_protecting_entries[branch] = self.node_protecting_entries[node]
  self.node_parents[node] = branch
  self.node_label_heads[node] = remainder
  self.node_label_lengths[node] -= common
  self.attach_node(node)
  branch
}

///|
fn PrefixIndex::attach_pages(
  self : PrefixIndex,
  terminal : Int,
  token_count : Int,
  page_size : Int,
  pages : PrefixPublishBuffer,
) -> Unit {
  let mut length = 0
  let mut node = terminal
  while node >= 0 {
    self.node_path_scratch[length] = node
    length += 1
    node = self.node_parents[node]
  }
  let mut position = 0
  for reverse in 0..<length {
    node = self.node_path_scratch[length - reverse - 1]
    let mut cell = self.node_label_heads[node]
    for offset in 0..<self.node_label_lengths[node] {
      position += 1
      if position % page_size == 0 && position <= token_count {
        let page_index = position / page_size - 1
        if self.token_page_cells[cell] < 0 {
          let page_cell = self.take_free_page()
          if page_cell < 0 {
            abort("prefix page preflight invariant")
          }
          try! self.page_values.write(page_cell, try! pages.page_at(page_index))
          self.token_page_cells[cell] = page_cell
          pages.adopted[page_index] = true
          self.page_count += 1
        }
      }
      if offset + 1 < self.node_label_lengths[node] {
        cell = self.token_next[cell]
      }
    }
  }
}

///|
fn PrefixIndex::insert_rooted(
  self : PrefixIndex,
  root : Int,
  tokens : PrefixTokenBuffer,
  pages : PrefixPublishBuffer,
  page_size : Int,
  priority : Int,
) -> PrefixEntryId raise PrefixIndexError {
  self.validate_publish_shape(tokens, pages, page_size, priority)
  let need = self.insertion_need(root, tokens, page_size)
  if need.duplicate {
    let mut parent = -1
    let mut position = 0
    while position < tokens.length() {
      parent = self.find_child(root, parent, tokens.token_at(position))
      position += self.node_label_lengths[parent]
    }
    raise DuplicatePrefix(
      PrefixEntryId(
        self.node_terminals[parent],
        self.entry_generations[self.node_terminals[parent]],
      ),
    )
  }
  if need.nodes > self.free_node_count() ||
    need.tokens > self.free_token_count() ||
    need.pages > self.free_page_count() {
    raise Exhausted
  }
  let entry = self.take_free_entry()
  if entry < 0 {
    raise Exhausted
  }
  let mut parent = -1
  let mut position = 0
  let mut terminal = -1
  while position < tokens.length() {
    let child = self.find_child(root, parent, tokens.token_at(position))
    if child < 0 {
      let length = tokens.length() - position
      let head = self.allocate_label(tokens, position, length)
      terminal = self.allocate_node(root, parent, head, length)
      position = tokens.length()
    } else {
      let mut cell = self.node_label_heads[child]
      let mut matched = 0
      while matched < self.node_label_lengths[child] &&
            position + matched < tokens.length() &&
            self.token_values[cell] == tokens.token_at(position + matched) {
        matched += 1
        if matched < self.node_label_lengths[child] {
          cell = self.token_next[cell]
        }
      }
      if matched < self.node_label_lengths[child] {
        let branch = self.split_node(child, matched)
        position += matched
        if position == tokens.length() {
          terminal = branch
        } else {
          let length = tokens.length() - position
          let head = self.allocate_label(tokens, position, length)
          terminal = self.allocate_node(root, branch, head, length)
          position = tokens.length()
        }
      } else {
        position += matched
        parent = child
        terminal = child
      }
    }
  }
  self.node_terminals[terminal] = entry
  let mut used = terminal
  while used >= 0 {
    self.node_usages[used] += 1
    used = self.node_parents[used]
  }
  for index in 0..<pages.length() {
    pages.adopted[index] = false
  }
  self.attach_pages(terminal, tokens.length(), page_size, pages)
  self.entry_states[entry] = ENTRY_USED
  self.entry_roots[entry] = root
  self.entry_terminals[entry] = terminal
  self.entry_token_counts[entry] = tokens.length()
  self.entry_page_counts[entry] = pages.length()
  self.entry_priorities[entry] = priority
  self.entry_active_references[entry] = 0
  self.entry_recencies[entry] = self.next_recency()
  self.root_entry_counts[root] += 1
  self.entry_count += 1
  self.propagate_protecting_entry(terminal, entry)
  self.eviction_heap_insert(entry)
  PrefixEntryId(entry, self.entry_generations[entry])
}
