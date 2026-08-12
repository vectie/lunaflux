///|
pub fn PrefixEntrySnapshot::id(self : PrefixEntrySnapshot) -> PrefixEntryId {
  self.id
}

///|
pub fn PrefixEntrySnapshot::token_count(self : PrefixEntrySnapshot) -> Int {
  self.token_count
}

///|
pub fn PrefixEntrySnapshot::page_count(self : PrefixEntrySnapshot) -> Int {
  self.page_count
}

///|
pub fn PrefixEntrySnapshot::active_reference_count(
  self : PrefixEntrySnapshot,
) -> Int {
  self.active_references
}

///|
pub fn PrefixEntrySnapshot::recency(self : PrefixEntrySnapshot) -> UInt64 {
  self.recency
}

///|
pub fn PrefixIndexSnapshot::entry_count(self : PrefixIndexSnapshot) -> Int {
  self.entry_count
}

///|
pub fn PrefixIndexSnapshot::retired_entry_count(
  self : PrefixIndexSnapshot,
) -> Int {
  self.retired_entry_count
}

///|
pub fn PrefixIndexSnapshot::node_count(self : PrefixIndexSnapshot) -> Int {
  self.node_count
}

///|
pub fn PrefixIndexSnapshot::page_count(self : PrefixIndexSnapshot) -> Int {
  self.page_count
}

///|
pub fn PrefixIndexSnapshot::snapshot_count(self : PrefixIndexSnapshot) -> Int {
  self.entries.length()
}

///|
pub fn PrefixIndexSnapshot::entry_at(
  self : PrefixIndexSnapshot,
  index : Int,
) -> PrefixEntrySnapshot raise PrefixIndexError {
  if index < 0 || index >= self.entries.length() {
    raise ResultIndexOutOfBounds
  }
  self.entries[index]
}

///|
pub fn PrefixIndex::debug_snapshot(self : PrefixIndex) -> PrefixIndexSnapshot {
  let entries : Array[PrefixEntrySnapshot] = []
  for index in 0..<self.entry_states.length() {
    if self.entry_states[index] == ENTRY_USED {
      entries.push({
        id: PrefixEntryId(index, self.entry_generations[index]),
        token_count: self.entry_token_counts[index],
        page_count: self.entry_page_counts[index],
        active_references: self.entry_active_references[index],
        recency: self.entry_recencies[index],
      })
    }
  }
  {
    entry_count: self.entry_count,
    retired_entry_count: self.retired_entry_count,
    node_count: self.node_count,
    page_count: self.page_count,
    entries: ReadOnlyArray::from_array(entries),
  }
}

///|
fn PrefixIndex::entry_path_contains(
  self : PrefixIndex,
  entry : Int,
  wanted : Int,
) -> Bool {
  let mut node = self.entry_terminals[entry]
  for _ in 0..<self.entry_token_counts[entry] {
    if node == wanted {
      return true
    }
    node = self.node_parents[node]
  }
  false
}

///|
pub fn PrefixIndex::debug_check_invariants(
  self : PrefixIndex,
) -> Unit raise PrefixIndexError {
  let mut used_entries = 0
  let mut retired_entries = 0
  for index in 0..<self.entry_states.length() {
    if self.entry_generations[index] <= 0 ||
      self.entry_generations[index] > self.limits.max_generation() {
      raise InvariantViolation(GenerationRange)
    }
    match self.entry_states[index] {
      0 => ()
      1 => used_entries = used_entries + 1
      2 => retired_entries = retired_entries + 1
      _ => raise InvariantViolation(CountPartition)
    }
  }
  if used_entries != self.entry_count ||
    retired_entries != self.retired_entry_count {
    raise InvariantViolation(CountPartition)
  }
  for root in 0..<self.roots.length() {
    let mut expected = 0
    for entry in 0..<self.entry_states.length() {
      if self.entry_states[entry] == ENTRY_USED &&
        self.entry_roots[entry] == root {
        expected = expected + 1
      }
    }
    if expected != self.root_entry_counts[root] ||
      (expected == 0) != (self.roots[root] is None) {
      raise InvariantViolation(RootCount)
    }
  }
  let mut used_nodes = 0
  for node in 0..<self.node_used.length() {
    if self.node_used[node] {
      used_nodes = used_nodes + 1
      let root = self.node_roots[node]
      let parent = self.node_parents[node]
      if root < 0 ||
        root >= self.roots.length() ||
        self.roots[root] is None ||
        (
          parent >= 0 &&
          (!self.node_used[parent] || self.node_roots[parent] != root)
        ) {
        raise InvariantViolation(NodeOwnership)
      }
      let mut expected_usage = 0
      for entry in 0..<self.entry_states.length() {
        if self.entry_states[entry] == ENTRY_USED &&
          self.entry_path_contains(entry, node) {
          expected_usage = expected_usage + 1
        }
      }
      if expected_usage <= 0 || self.node_usages[node] != expected_usage {
        raise InvariantViolation(NodeUsage)
      }
      let terminal = self.node_terminals[node]
      if terminal >= 0 &&
        (
          terminal >= self.entry_states.length() ||
          self.entry_states[terminal] != ENTRY_USED ||
          self.entry_terminals[terminal] != node
        ) {
        raise InvariantViolation(TerminalLink)
      }
    } else if self.node_usages[node] != 0 || self.node_terminals[node] != -1 {
      raise InvariantViolation(NodeUsage)
    }
  }
  if used_nodes != self.node_count {
    raise InvariantViolation(CountPartition)
  }
  let mut referenced_pages = 0
  for entry in 0..<self.entry_states.length() {
    if self.entry_states[entry] == ENTRY_USED {
      let root = self.entry_roots[entry]
      guard self.roots[root] is Some(identity) else {
        raise InvariantViolation(EntryShape)
      }
      if self.entry_token_counts[entry] !=
        self.entry_page_counts[entry] * identity.layout().tokens_per_page() ||
        self.node_terminals[self.entry_terminals[entry]] != entry {
        raise InvariantViolation(EntryShape)
      }
      let mut cell = self.entry_page_heads[entry]
      for _ in 0..<self.entry_page_counts[entry] {
        if cell < 0 ||
          cell >= self.page_used.length() ||
          !self.page_used[cell] ||
          self.page_values[cell] is None {
          raise InvariantViolation(PageChain)
        }
        referenced_pages = referenced_pages + 1
        cell = self.page_next[cell]
      }
      if cell != -1 {
        raise InvariantViolation(PageChain)
      }
    }
  }
  let mut used_pages = 0
  for page in 0..<self.page_used.length() {
    if self.page_used[page] {
      used_pages = used_pages + 1
      let mut owners = 0
      for entry in 0..<self.entry_states.length() {
        if self.entry_states[entry] == ENTRY_USED {
          let mut cell = self.entry_page_heads[entry]
          for _ in 0..<self.entry_page_counts[entry] {
            if cell == page {
              owners = owners + 1
            }
            cell = self.page_next[cell]
          }
        }
      }
      if owners != 1 {
        raise InvariantViolation(PageChain)
      }
    } else if self.page_values[page] is Some(_) || self.page_next[page] != -1 {
      raise InvariantViolation(PageChain)
    }
  }
  if used_pages != self.page_count || referenced_pages != self.page_count {
    raise InvariantViolation(CountPartition)
  }
}
