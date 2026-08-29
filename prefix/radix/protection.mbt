///|
fn PrefixIndex::entry_protection_precedes(
  self : PrefixIndex,
  left : Int,
  right : Int,
) -> Bool {
  right < 0 ||
  self.entry_token_counts[left] < self.entry_token_counts[right] ||
  (
    self.entry_token_counts[left] == self.entry_token_counts[right] &&
    left < right
  )
}

///|
fn PrefixIndex::propagate_protecting_entry(
  self : PrefixIndex,
  terminal : Int,
  entry : Int,
) -> Unit {
  let mut node = terminal
  while node >= 0 {
    if self.entry_protection_precedes(entry, self.node_protecting_entries[node]) {
      self.node_protecting_entries[node] = entry
    }
    node = self.node_parents[node]
  }
}

///|
fn PrefixIndex::refresh_node_protector(self : PrefixIndex, node : Int) -> Unit {
  let mut selected = self.node_terminals[node]
  let mut child = self.node_child_heads[node]
  while child >= 0 {
    self.compaction_incident_edges += 1
    let candidate = self.node_protecting_entries[child]
    if candidate >= 0 && self.entry_protection_precedes(candidate, selected) {
      selected = candidate
    }
    child = self.node_sibling_next[child]
  }
  self.node_protecting_entries[node] = selected
}

///|
fn PrefixIndex::protecting_entry(self : PrefixIndex, node : Int) -> Int {
  self.node_protecting_entries[node]
}
