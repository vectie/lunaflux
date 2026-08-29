///|
fn count_free_link(
  head : Int,
  links : FixedArray[Int],
  target : Int,
) -> Int raise PrefixIndexError {
  let mut count = 0
  let mut cursor = head
  let mut steps = 0
  while cursor >= 0 {
    if cursor >= links.length() || steps >= links.length() {
      raise InvariantViolation(CountPartition)
    }
    if cursor == target {
      count += 1
    }
    cursor = links[cursor]
    steps += 1
  }
  count
}

///|
fn PrefixIndex::debug_check_free_stacks(
  self : PrefixIndex,
) -> Unit raise PrefixIndexError {
  for root in 0..<self.root_used.length() {
    let membership = count_free_link(
      self.root_free_head,
      self.root_free_next,
      root,
    )
    if membership != (if self.root_used[root] { 0 } else { 1 }) {
      raise InvariantViolation(RootCount)
    }
  }
  for entry in 0..<self.entry_states.length() {
    let membership = count_free_link(
      self.entry_free_head,
      self.entry_free_next,
      entry,
    )
    if membership !=
      (if self.entry_states[entry] == ENTRY_FREE { 1 } else { 0 }) {
      raise InvariantViolation(CountPartition)
    }
  }
  for node in 0..<self.node_used.length() {
    let membership = count_free_link(
      self.node_free_head,
      self.node_free_next,
      node,
    )
    if membership != (if self.node_used[node] { 0 } else { 1 }) {
      raise InvariantViolation(CountPartition)
    }
  }
  for token in 0..<self.token_used.length() {
    let membership = count_free_link(
      self.token_free_head,
      self.token_free_next,
      token,
    )
    if membership != (if self.token_used[token] { 0 } else { 1 }) {
      raise InvariantViolation(CountPartition)
    }
  }
  for page in 0..<self.limits.page_capacity() {
    let membership = count_free_link(
      self.page_free_head,
      self.page_free_next,
      page,
    )
    let occupied = try! self.page_values.is_occupied(page)
    if membership != (if occupied { 0 } else { 1 }) {
      raise InvariantViolation(CountPartition)
    }
  }
}

///|
fn PrefixIndex::debug_check_root_index(
  self : PrefixIndex,
) -> Unit raise PrefixIndexError {
  for root in 0..<self.root_used.length() {
    let edge = self.root_index_slots[root]
    if self.root_used[root] {
      if edge < 0 || self.root_index_payloads[edge] != root {
        raise InvariantViolation(RootCount)
      }
    } else if edge != -1 {
      raise InvariantViolation(RootCount)
    }
  }
  for edge in 0..<self.root_index_payloads.length() {
    let root = self.root_index_payloads[edge]
    let free_membership = count_free_link(
      self.root_index_free_head,
      self.root_index_free_next,
      edge,
    )
    if root < 0 {
      if free_membership != 1 ||
        self.root_index_left[edge] != -1 ||
        self.root_index_right[edge] != -1 ||
        self.root_index_parents[edge] != -1 ||
        self.root_index_heights[edge] != 0 {
        raise InvariantViolation(RootCount)
      }
      continue
    }
    if free_membership != 0 ||
      root >= self.root_used.length() ||
      !self.root_used[root] ||
      self.root_index_slots[root] != edge {
      raise InvariantViolation(RootCount)
    }
    let left = self.root_index_left[edge]
    let right = self.root_index_right[edge]
    if left >= 0 &&
      (
        self.root_index_parents[left] != edge ||
        self.compare_root_fixed(self.root_index_payloads[left], root) >= 0
      ) {
      raise InvariantViolation(RootCount)
    }
    if right >= 0 &&
      (
        self.root_index_parents[right] != edge ||
        self.compare_root_fixed(self.root_index_payloads[right], root) <= 0
      ) {
      raise InvariantViolation(RootCount)
    }
    let left_height = self.root_index_height(left)
    let right_height = self.root_index_height(right)
    if self.root_index_heights[edge] != 1 + Int::max(left_height, right_height) ||
      left_height - right_height < -1 ||
      left_height - right_height > 1 {
      raise InvariantViolation(RootCount)
    }
    let mut ancestor = edge
    let mut steps = 0
    while self.root_index_parents[ancestor] >= 0 {
      ancestor = self.root_index_parents[ancestor]
      steps += 1
      if steps >= self.root_index_payloads.length() {
        raise InvariantViolation(RootCount)
      }
    }
    if ancestor != self.root_index_root {
      raise InvariantViolation(RootCount)
    }
  }
  if self.root_index_root >= 0 &&
    self.root_index_parents[self.root_index_root] != -1 {
    raise InvariantViolation(RootCount)
  }
}

///|
fn PrefixIndex::debug_check_child_lists(
  self : PrefixIndex,
) -> Unit raise PrefixIndexError {
  for node in 0..<self.node_used.length() {
    if !self.node_used[node] {
      continue
    }
    let parent = self.node_parents[node]
    let mut cursor = if parent < 0 {
      self.root_child_heads[self.node_roots[node]]
    } else {
      self.node_child_heads[parent]
    }
    let mut membership = 0
    let mut previous = -1
    let mut steps = 0
    while cursor >= 0 {
      if cursor >= self.node_used.length() ||
        !self.node_used[cursor] ||
        self.node_sibling_prev[cursor] != previous ||
        self.node_parents[cursor] != parent ||
        self.node_roots[cursor] != self.node_roots[node] ||
        steps >= self.node_used.length() {
        raise InvariantViolation(NodeOwnership)
      }
      if cursor == node {
        membership += 1
      }
      previous = cursor
      cursor = self.node_sibling_next[cursor]
      steps += 1
    }
    if membership != 1 {
      raise InvariantViolation(NodeOwnership)
    }
  }
  for root in 0..<self.root_used.length() {
    let mut count = 0
    let mut child = self.root_child_heads[root]
    while child >= 0 {
      count += 1
      child = self.node_sibling_next[child]
    }
    if count != self.root_child_counts[root] {
      raise InvariantViolation(NodeOwnership)
    }
  }
  for parent in 0..<self.node_used.length() {
    if self.node_used[parent] {
      let mut count = 0
      let mut child = self.node_child_heads[parent]
      while child >= 0 {
        count += 1
        child = self.node_sibling_next[child]
      }
      if count != self.node_child_counts[parent] {
        raise InvariantViolation(NodeOwnership)
      }
    }
  }
}

///|
fn PrefixIndex::debug_check_child_index(
  self : PrefixIndex,
) -> Unit raise PrefixIndexError {
  for node in 0..<self.node_used.length() {
    let edge = self.child_index_slots[node]
    if self.node_used[node] {
      if edge < 0 || self.child_index_payloads[edge] != node {
        raise InvariantViolation(NodeOwnership)
      }
    } else if edge != -1 {
      raise InvariantViolation(NodeOwnership)
    }
  }
  for edge in 0..<self.child_index_payloads.length() {
    let node = self.child_index_payloads[edge]
    let membership = count_free_link(
      self.child_index_free_head,
      self.child_index_free_next,
      edge,
    )
    if node < 0 {
      if membership != 1 ||
        self.child_index_left[edge] != -1 ||
        self.child_index_right[edge] != -1 ||
        self.child_index_parents[edge] != -1 ||
        self.child_index_heights[edge] != 0 {
        raise InvariantViolation(NodeOwnership)
      }
      continue
    }
    if membership != 0 ||
      !self.node_used[node] ||
      self.child_index_slots[node] != edge {
      raise InvariantViolation(NodeOwnership)
    }
    let left = self.child_index_left[edge]
    let right = self.child_index_right[edge]
    if left >= 0 &&
      (
        self.child_index_parents[left] != edge ||
        self.compare_child_nodes(self.child_index_payloads[left], node) >= 0
      ) {
      raise InvariantViolation(NodeOwnership)
    }
    if right >= 0 &&
      (
        self.child_index_parents[right] != edge ||
        self.compare_child_nodes(self.child_index_payloads[right], node) <= 0
      ) {
      raise InvariantViolation(NodeOwnership)
    }
    let left_height = self.child_index_height(left)
    let right_height = self.child_index_height(right)
    if self.child_index_heights[edge] != 1 + Int::max(left_height, right_height) ||
      left_height - right_height < -1 ||
      left_height - right_height > 1 {
      raise InvariantViolation(NodeOwnership)
    }
    let mut ancestor = edge
    let mut steps = 0
    while self.child_index_parents[ancestor] >= 0 {
      ancestor = self.child_index_parents[ancestor]
      steps += 1
      if steps >= self.child_index_payloads.length() {
        raise InvariantViolation(NodeOwnership)
      }
    }
    if ancestor != self.child_index_root {
      raise InvariantViolation(NodeOwnership)
    }
  }
}

///|
fn PrefixIndex::debug_check_eviction_heap(
  self : PrefixIndex,
) -> Unit raise PrefixIndexError {
  let mut expected = 0
  for entry in 0..<self.entry_states.length() {
    let eligible = self.entry_states[entry] == ENTRY_USED &&
      self.entry_active_references[entry] == 0
    let position = self.eviction_heap_positions[entry]
    if eligible {
      expected += 1
      if position < 0 ||
        position >= self.eviction_heap_count ||
        self.eviction_heap[position] != entry {
        raise InvariantViolation(CountPartition)
      }
    } else if position != -1 {
      raise InvariantViolation(CountPartition)
    }
  }
  if expected != self.eviction_heap_count {
    raise InvariantViolation(CountPartition)
  }
  for position in 0..<self.eviction_heap_count {
    let entry = self.eviction_heap[position]
    let left = position * 2 + 1
    let right = left + 1
    if left < self.eviction_heap_count &&
      self.eviction_precedes(self.eviction_heap[left], entry) {
      raise InvariantViolation(CountPartition)
    }
    if right < self.eviction_heap_count &&
      self.eviction_precedes(self.eviction_heap[right], entry) {
      raise InvariantViolation(CountPartition)
    }
  }
}

///|
fn PrefixIndex::debug_check_protectors(
  self : PrefixIndex,
) -> Unit raise PrefixIndexError {
  for node in 0..<self.node_used.length() {
    if !self.node_used[node] {
      continue
    }
    let mut expected = -1
    for entry in 0..<self.entry_states.length() {
      if self.entry_states[entry] != ENTRY_USED {
        continue
      }
      let mut ancestor = self.entry_terminals[entry]
      while ancestor >= 0 {
        if ancestor == node {
          if self.entry_protection_precedes(entry, expected) {
            expected = entry
          }
          break
        }
        ancestor = self.node_parents[ancestor]
      }
    }
    if self.node_protecting_entries[node] != expected {
      raise InvariantViolation(NodeUsage)
    }
  }
}

///|
fn PrefixIndex::debug_check_operation_counters(
  self : PrefixIndex,
) -> Unit raise PrefixIndexError {
  if self.child_lookup_comparisons < 0 ||
    self.root_lookup_comparisons < 0 ||
    self.duplicate_page_comparisons < 0 ||
    self.duplicate_page_swaps < 0 ||
    self.recency_normalization_comparisons < 0 ||
    self.recency_normalization_swaps < 0 ||
    self.free_stack_touches < 0 ||
    self.eviction_heap_touches < 0 ||
    self.compaction_node_visits < 0 ||
    self.compaction_incident_edges < 0 {
    raise InvariantViolation(CountPartition)
  }
  for index in 0..<self.root_scope_scratch.length() {
    if self.root_scope_scratch[index] != b'\x00' {
      raise InvariantViolation(RootCount)
    }
  }
  for index in 0..<self.recency_entry_scratch.length() {
    if self.recency_entry_scratch[index] != -1 {
      raise InvariantViolation(CountPartition)
    }
  }
  for index in 0..<self.limits.page_capacity() {
    if (try! self.duplicate_page_scratch.is_occupied(index)) {
      raise InvariantViolation(PageChain)
    }
  }
}
