///|
fn PrefixIndex::compare_child_key(
  self : PrefixIndex,
  root : Int,
  parent : Int,
  token : Int,
  node : Int,
) -> Int {
  let mut order = compare_int(root, self.node_roots[node])
  if order != 0 {
    return order
  }
  order = compare_int(parent, self.node_parents[node])
  if order != 0 {
    return order
  }
  compare_int(token, self.node_first_token(node))
}

///|
fn PrefixIndex::compare_child_nodes(
  self : PrefixIndex,
  left : Int,
  right : Int,
) -> Int {
  self.compare_child_key(
    self.node_roots[left],
    self.node_parents[left],
    self.node_first_token(left),
    right,
  )
}

///|
fn PrefixIndex::child_index_height(self : PrefixIndex, edge : Int) -> Int {
  if edge < 0 {
    0
  } else {
    self.child_index_heights[edge]
  }
}

///|
fn PrefixIndex::child_index_update_height(
  self : PrefixIndex,
  edge : Int,
) -> Unit {
  self.child_index_heights[edge] = 1 +
    Int::max(
      self.child_index_height(self.child_index_left[edge]),
      self.child_index_height(self.child_index_right[edge]),
    )
}

///|
fn PrefixIndex::child_index_replace_parent(
  self : PrefixIndex,
  old_edge : Int,
  new_edge : Int,
) -> Unit {
  let parent = self.child_index_parents[old_edge]
  if parent < 0 {
    self.child_index_root = new_edge
  } else if self.child_index_left[parent] == old_edge {
    self.child_index_left[parent] = new_edge
  } else {
    self.child_index_right[parent] = new_edge
  }
  if new_edge >= 0 {
    self.child_index_parents[new_edge] = parent
  }
}

///|
fn PrefixIndex::child_index_rotate_left(self : PrefixIndex, edge : Int) -> Int {
  let pivot = self.child_index_right[edge]
  let middle = self.child_index_left[pivot]
  self.child_index_replace_parent(edge, pivot)
  self.child_index_left[pivot] = edge
  self.child_index_parents[edge] = pivot
  self.child_index_right[edge] = middle
  if middle >= 0 {
    self.child_index_parents[middle] = edge
  }
  self.child_index_update_height(edge)
  self.child_index_update_height(pivot)
  pivot
}

///|
fn PrefixIndex::child_index_rotate_right(self : PrefixIndex, edge : Int) -> Int {
  let pivot = self.child_index_left[edge]
  let middle = self.child_index_right[pivot]
  self.child_index_replace_parent(edge, pivot)
  self.child_index_right[pivot] = edge
  self.child_index_parents[edge] = pivot
  self.child_index_left[edge] = middle
  if middle >= 0 {
    self.child_index_parents[middle] = edge
  }
  self.child_index_update_height(edge)
  self.child_index_update_height(pivot)
  pivot
}

///|
fn PrefixIndex::child_index_rebalance(self : PrefixIndex, start : Int) -> Unit {
  let mut edge = start
  while edge >= 0 {
    self.child_index_update_height(edge)
    let balance = self.child_index_height(self.child_index_left[edge]) -
      self.child_index_height(self.child_index_right[edge])
    let balanced = if balance > 1 {
      let left = self.child_index_left[edge]
      if self.child_index_height(self.child_index_left[left]) <
        self.child_index_height(self.child_index_right[left]) {
        ignore(self.child_index_rotate_left(left))
      }
      self.child_index_rotate_right(edge)
    } else if balance < -1 {
      let right = self.child_index_right[edge]
      if self.child_index_height(self.child_index_right[right]) <
        self.child_index_height(self.child_index_left[right]) {
        ignore(self.child_index_rotate_right(right))
      }
      self.child_index_rotate_left(edge)
    } else {
      edge
    }
    edge = self.child_index_parents[balanced]
  }
}

///|
fn PrefixIndex::child_index_insert(self : PrefixIndex, node : Int) -> Unit {
  let edge = self.child_index_free_head
  if edge < 0 {
    abort("prefix child-index preflight invariant")
  }
  self.child_index_free_head = self.child_index_free_next[edge]
  self.child_index_free_next[edge] = -1
  self.free_stack_touches += 1
  self.child_index_payloads[edge] = node
  self.child_index_left[edge] = -1
  self.child_index_right[edge] = -1
  self.child_index_parents[edge] = -1
  self.child_index_heights[edge] = 1
  self.child_index_slots[node] = edge
  if self.child_index_root < 0 {
    self.child_index_root = edge
    return
  }
  let mut current = self.child_index_root
  for ;; {
    let order = self.compare_child_nodes(
      node,
      self.child_index_payloads[current],
    )
    if order == 0 {
      abort("prefix duplicate child-index invariant")
    }
    let next = if order < 0 {
      self.child_index_left[current]
    } else {
      self.child_index_right[current]
    }
    if next >= 0 {
      current = next
    } else {
      self.child_index_parents[edge] = current
      if order < 0 {
        self.child_index_left[current] = edge
      } else {
        self.child_index_right[current] = edge
      }
      self.child_index_rebalance(current)
      return
    }
  }
}

///|
fn PrefixIndex::child_index_remove(self : PrefixIndex, node : Int) -> Unit {
  let mut edge = self.child_index_slots[node]
  if edge < 0 {
    abort("prefix child-index removal invariant")
  }
  if self.child_index_left[edge] >= 0 && self.child_index_right[edge] >= 0 {
    let mut successor = self.child_index_right[edge]
    while self.child_index_left[successor] >= 0 {
      successor = self.child_index_left[successor]
    }
    let successor_node = self.child_index_payloads[successor]
    self.child_index_payloads[edge] = successor_node
    self.child_index_slots[successor_node] = edge
    self.child_index_payloads[successor] = node
    self.child_index_slots[node] = successor
    edge = successor
  }
  let child = if self.child_index_left[edge] >= 0 {
    self.child_index_left[edge]
  } else {
    self.child_index_right[edge]
  }
  let parent = self.child_index_parents[edge]
  self.child_index_replace_parent(edge, child)
  self.child_index_slots[node] = -1
  self.child_index_payloads[edge] = -1
  self.child_index_left[edge] = -1
  self.child_index_right[edge] = -1
  self.child_index_parents[edge] = -1
  self.child_index_heights[edge] = 0
  self.child_index_free_next[edge] = self.child_index_free_head
  self.child_index_free_head = edge
  self.free_stack_touches += 1
  self.child_index_rebalance(parent)
}

///|
fn PrefixIndex::link_child(self : PrefixIndex, node : Int) -> Unit {
  let parent = self.node_parents[node]
  let head = if parent < 0 {
    self.root_child_heads[self.node_roots[node]]
  } else {
    self.node_child_heads[parent]
  }
  self.node_sibling_prev[node] = -1
  self.node_sibling_next[node] = head
  if head >= 0 {
    self.node_sibling_prev[head] = node
  }
  if parent < 0 {
    let root = self.node_roots[node]
    self.root_child_heads[root] = node
    self.root_child_counts[root] += 1
  } else {
    self.node_child_heads[parent] = node
    self.node_child_counts[parent] += 1
  }
}

///|
fn PrefixIndex::unlink_child(self : PrefixIndex, node : Int) -> Unit {
  let parent = self.node_parents[node]
  let previous = self.node_sibling_prev[node]
  let next = self.node_sibling_next[node]
  if previous >= 0 {
    self.node_sibling_next[previous] = next
  } else if parent < 0 {
    self.root_child_heads[self.node_roots[node]] = next
  } else {
    self.node_child_heads[parent] = next
  }
  if next >= 0 {
    self.node_sibling_prev[next] = previous
  }
  if parent < 0 {
    self.root_child_counts[self.node_roots[node]] -= 1
  } else {
    self.node_child_counts[parent] -= 1
  }
  self.node_sibling_prev[node] = -1
  self.node_sibling_next[node] = -1
}

///|
fn PrefixIndex::attach_node(self : PrefixIndex, node : Int) -> Unit {
  self.link_child(node)
  self.child_index_insert(node)
}

///|
fn PrefixIndex::detach_node(self : PrefixIndex, node : Int) -> Unit {
  self.child_index_remove(node)
  self.unlink_child(node)
}

///|
fn PrefixIndex::find_child(
  self : PrefixIndex,
  root : Int,
  parent : Int,
  token : Int,
) -> Int {
  let mut edge = self.child_index_root
  while edge >= 0 {
    self.child_lookup_comparisons += 1
    let node = self.child_index_payloads[edge]
    let order = self.compare_child_key(root, parent, token, node)
    if order == 0 {
      return node
    }
    edge = if order < 0 {
      self.child_index_left[edge]
    } else {
      self.child_index_right[edge]
    }
  }
  -1
}

///|
fn PrefixIndex::single_child(self : PrefixIndex, parent : Int) -> Int {
  if self.node_child_counts[parent] == 1 {
    self.node_child_heads[parent]
  } else if self.node_child_counts[parent] == 0 {
    -1
  } else {
    -2
  }
}

///|
fn PrefixIndex::take_free_node(self : PrefixIndex) -> Int {
  let node = self.node_free_head
  if node < 0 {
    return -1
  }
  self.node_free_head = self.node_free_next[node]
  self.node_free_next[node] = -1
  self.free_stack_touches += 1
  node
}

///|
fn PrefixIndex::release_node_slot(self : PrefixIndex, node : Int) -> Unit {
  self.node_free_next[node] = self.node_free_head
  self.node_free_head = node
  self.free_stack_touches += 1
}
