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
pub fn PrefixIndex::debug_check_invariants(
  self : PrefixIndex,
) -> Unit raise PrefixIndexError {
  self.debug_check_free_stacks()
  self.debug_check_root_index()
  self.debug_check_child_lists()
  self.debug_check_child_index()
  self.debug_check_eviction_heap()
  self.debug_check_protectors()
  self.debug_check_operation_counters()
  let mut entries = 0
  let mut retired = 0
  for root in 0..<self.root_used.length() {
    let mut rooted_entries = 0
    for entry in 0..<self.entry_states.length() {
      if self.entry_states[entry] == ENTRY_USED &&
        self.entry_roots[entry] == root {
        rooted_entries += 1
      }
    }
    if self.root_used[root] {
      if rooted_entries == 0 ||
        rooted_entries != self.root_entry_counts[root] ||
        self.root_models[root] is None ||
        self.root_tokenizers[root] is None ||
        self.root_layouts[root] is None ||
        self.root_scope_lengths[root] < 0 ||
        self.root_scope_lengths[root] > self.limits.max_scope_bytes() {
        raise InvariantViolation(RootCount)
      }
    } else if rooted_entries != 0 ||
      self.root_entry_counts[root] != 0 ||
      self.root_models[root] is Some(_) ||
      self.root_tokenizers[root] is Some(_) ||
      self.root_layouts[root] is Some(_) ||
      self.root_scope_lengths[root] != 0 {
      raise InvariantViolation(RootCount)
    }
  }
  for entry in 0..<self.entry_states.length() {
    match self.entry_states[entry] {
      ENTRY_FREE => ()
      ENTRY_USED => {
        entries += 1
        let terminal = self.entry_terminals[entry]
        if terminal < 0 ||
          !self.node_used[terminal] ||
          self.node_terminals[terminal] != entry ||
          self.entry_token_counts[entry] !=
          self.entry_page_counts[entry] *
          self.root_layouts[self.entry_roots[entry]].unwrap().tokens_per_page() {
          raise InvariantViolation(EntryShape)
        }
        let mut node = terminal
        let mut path_tokens = 0
        let mut path_pages = 0
        while node >= 0 {
          if self.node_roots[node] != self.entry_roots[entry] {
            raise InvariantViolation(NodeOwnership)
          }
          path_tokens += self.node_label_lengths[node]
          let mut cell = self.node_label_heads[node]
          for offset in 0..<self.node_label_lengths[node] {
            if self.token_page_cells[cell] >= 0 {
              path_pages += 1
            }
            if offset + 1 < self.node_label_lengths[node] {
              cell = self.token_next[cell]
            }
          }
          node = self.node_parents[node]
        }
        if path_tokens != self.entry_token_counts[entry] ||
          path_pages != self.entry_page_counts[entry] {
          raise InvariantViolation(EntryShape)
        }
      }
      ENTRY_RETIRED => retired += 1
      _ => raise InvariantViolation(CountPartition)
    }
  }
  if entries != self.entry_count || retired != self.retired_entry_count {
    raise InvariantViolation(CountPartition)
  }
  let mut nodes = 0
  let mut tokens = 0
  for node in 0..<self.node_used.length() {
    if self.node_used[node] {
      nodes += 1
      if self.node_label_lengths[node] <= 0 ||
        self.node_label_heads[node] < 0 ||
        self.node_usages[node] <= 0 {
        raise InvariantViolation(CompressedShape)
      }
      if self.node_terminals[node] < 0 && self.single_child(node) >= 0 {
        raise InvariantViolation(CompressedShape)
      }
      let mut expected_usage = 0
      for entry in 0..<self.entry_states.length() {
        if self.entry_states[entry] == ENTRY_USED {
          let mut ancestor = self.entry_terminals[entry]
          while ancestor >= 0 {
            if ancestor == node {
              expected_usage += 1
              break
            }
            ancestor = self.node_parents[ancestor]
          }
        }
      }
      if expected_usage != self.node_usages[node] {
        raise InvariantViolation(NodeUsage)
      }
      let mut cell = self.node_label_heads[node]
      for offset in 0..<self.node_label_lengths[node] {
        if cell < 0 || !self.token_used[cell] {
          raise InvariantViolation(NodeOwnership)
        }
        tokens += 1
        if offset + 1 < self.node_label_lengths[node] {
          cell = self.token_next[cell]
        }
      }
      if cell != self.node_label_tails[node] || self.token_next[cell] != -1 {
        raise InvariantViolation(NodeOwnership)
      }
    }
  }
  if nodes != self.node_count || tokens != self.token_count {
    raise InvariantViolation(CountPartition)
  }
  let mut pages = 0
  for cell in 0..<self.limits.page_capacity() {
    if (try! self.page_values.is_occupied(cell)) {
      let mut owners = 0
      for token in 0..<self.token_used.length() {
        if self.token_used[token] && self.token_page_cells[token] == cell {
          owners += 1
        }
      }
      if owners != 1 {
        raise InvariantViolation(PageChain)
      }
      pages += 1
    }
  }
  if pages != self.page_count {
    raise InvariantViolation(CountPartition)
  }
  if self.publish_plan_active && self.eviction_plan_active {
    raise InvariantViolation(CompressedShape)
  }
  if self.publish_plan_active {
    if self.publish_plan_generation == 0UL ||
      self.publish_plan_token_count <= 0 ||
      self.publish_plan_page_count <= 0 {
      raise InvariantViolation(CompressedShape)
    }
    for index in 0..<self.publish_plan_page_count {
      if !(try! self.publish_page_scratch.is_occupied(index)) {
        raise InvariantViolation(PageChain)
      }
    }
  } else if self.publish_plan_root != -1 ||
    self.publish_plan_token_count != 0 ||
    self.publish_plan_page_count != 0 ||
    self.publish_plan_priority != 0 {
    raise InvariantViolation(CompressedShape)
  }
  if self.eviction_plan_active {
    if self.eviction_plan_generation == 0UL ||
      self.eviction_plan_entry_index < 0 ||
      self.eviction_plan_entry_generation <= 0 ||
      self.eviction_plan_page_count < 0 {
      raise InvariantViolation(CompressedShape)
    }
    for index in 0..<self.eviction_plan_page_count {
      if !(try! self.eviction_page_scratch.is_occupied(index)) {
        raise InvariantViolation(PageChain)
      }
    }
  } else if self.eviction_plan_entry_index != -1 ||
    self.eviction_plan_entry_generation != 0 ||
    self.eviction_plan_page_count != 0 {
    raise InvariantViolation(CompressedShape)
  }
}
