///|
fn compare_int(left : Int, right : Int) -> Int {
  if left < right {
    -1
  } else if left > right {
    1
  } else {
    0
  }
}

///|
fn compare_string(left : String, right : String) -> Int {
  let common = Int::min(left.length(), right.length())
  for index in 0..<common {
    if left[index] < right[index] {
      return -1
    }
    if left[index] > right[index] {
      return 1
    }
  }
  compare_int(left.length(), right.length())
}

///|
fn PrefixIndex::compare_root_fixed(
  self : PrefixIndex,
  left : Int,
  right : Int,
) -> Int {
  let left_model = self.root_models[left].unwrap()
  let right_model = self.root_models[right].unwrap()
  let mut order = compare_string(
    left_model.content().as_hex(),
    right_model.content().as_hex(),
  )
  if order != 0 {
    return order
  }
  order = compare_string(
    left_model.plan().as_hex(),
    right_model.plan().as_hex(),
  )
  if order != 0 {
    return order
  }
  order = compare_string(
    self.root_tokenizers[left].unwrap().as_string(),
    self.root_tokenizers[right].unwrap().as_string(),
  )
  if order != 0 {
    return order
  }
  let left_layout = self.root_layouts[left].unwrap()
  let right_layout = self.root_layouts[right].unwrap()
  order = compare_int(left_layout.version(), right_layout.version())
  if order != 0 {
    return order
  }
  order = compare_int(
    left_layout.tokens_per_page(),
    right_layout.tokens_per_page(),
  )
  if order != 0 {
    return order
  }
  let common = Int::min(
    self.root_scope_lengths[left],
    self.root_scope_lengths[right],
  )
  for index in 0..<common {
    let left_byte = self.root_scope_byte(left, index)
    let right_byte = self.root_scope_byte(right, index)
    if left_byte < right_byte {
      return -1
    }
    if left_byte > right_byte {
      return 1
    }
  }
  compare_int(self.root_scope_lengths[left], self.root_scope_lengths[right])
}

///|
fn PrefixIndex::compare_request_root(
  self : PrefixIndex,
  key : PrefixRequestKey,
  root : Int,
) -> Int raise PrefixIndexError {
  if !key.is_live() {
    raise StaleKey
  }
  let model = self.root_models[root].unwrap()
  let mut order = compare_string(
    key.model_identity.content().as_hex(),
    model.content().as_hex(),
  )
  if order != 0 {
    return order
  }
  order = compare_string(
    key.model_identity.plan().as_hex(),
    model.plan().as_hex(),
  )
  if order != 0 {
    return order
  }
  order = compare_string(
    key.tokenizer.as_string(),
    self.root_tokenizers[root].unwrap().as_string(),
  )
  if order != 0 {
    return order
  }
  let layout = self.root_layouts[root].unwrap()
  order = compare_int(key.layout.version(), layout.version())
  if order != 0 {
    return order
  }
  order = compare_int(key.layout.tokens_per_page(), layout.tokens_per_page())
  if order != 0 {
    return order
  }
  let length = key.semantic.scope_length() catch { _ => raise StaleKey }
  let common = Int::min(length, self.root_scope_lengths[root])
  for index in 0..<common {
    let left = key.semantic.scope_byte_at(index) catch { _ => raise StaleKey }
    let right = self.root_scope_byte(root, index)
    if left < right {
      return -1
    }
    if left > right {
      return 1
    }
  }
  compare_int(length, self.root_scope_lengths[root])
}

///|
fn PrefixIndex::root_index_height(self : PrefixIndex, edge : Int) -> Int {
  if edge < 0 {
    0
  } else {
    self.root_index_heights[edge]
  }
}

///|
fn PrefixIndex::root_index_update_height(
  self : PrefixIndex,
  edge : Int,
) -> Unit {
  self.root_index_heights[edge] = 1 +
    Int::max(
      self.root_index_height(self.root_index_left[edge]),
      self.root_index_height(self.root_index_right[edge]),
    )
}

///|
fn PrefixIndex::root_index_replace_parent(
  self : PrefixIndex,
  old_edge : Int,
  new_edge : Int,
) -> Unit {
  let parent = self.root_index_parents[old_edge]
  if parent < 0 {
    self.root_index_root = new_edge
  } else if self.root_index_left[parent] == old_edge {
    self.root_index_left[parent] = new_edge
  } else {
    self.root_index_right[parent] = new_edge
  }
  if new_edge >= 0 {
    self.root_index_parents[new_edge] = parent
  }
}

///|
fn PrefixIndex::root_index_rotate_left(self : PrefixIndex, edge : Int) -> Int {
  let pivot = self.root_index_right[edge]
  let middle = self.root_index_left[pivot]
  self.root_index_replace_parent(edge, pivot)
  self.root_index_left[pivot] = edge
  self.root_index_parents[edge] = pivot
  self.root_index_right[edge] = middle
  if middle >= 0 {
    self.root_index_parents[middle] = edge
  }
  self.root_index_update_height(edge)
  self.root_index_update_height(pivot)
  pivot
}

///|
fn PrefixIndex::root_index_rotate_right(self : PrefixIndex, edge : Int) -> Int {
  let pivot = self.root_index_left[edge]
  let middle = self.root_index_right[pivot]
  self.root_index_replace_parent(edge, pivot)
  self.root_index_right[pivot] = edge
  self.root_index_parents[edge] = pivot
  self.root_index_left[edge] = middle
  if middle >= 0 {
    self.root_index_parents[middle] = edge
  }
  self.root_index_update_height(edge)
  self.root_index_update_height(pivot)
  pivot
}

///|
fn PrefixIndex::root_index_rebalance(self : PrefixIndex, start : Int) -> Unit {
  let mut edge = start
  while edge >= 0 {
    self.root_index_update_height(edge)
    let balance = self.root_index_height(self.root_index_left[edge]) -
      self.root_index_height(self.root_index_right[edge])
    let balanced = if balance > 1 {
      let left = self.root_index_left[edge]
      if self.root_index_height(self.root_index_left[left]) <
        self.root_index_height(self.root_index_right[left]) {
        ignore(self.root_index_rotate_left(left))
      }
      self.root_index_rotate_right(edge)
    } else if balance < -1 {
      let right = self.root_index_right[edge]
      if self.root_index_height(self.root_index_right[right]) <
        self.root_index_height(self.root_index_left[right]) {
        ignore(self.root_index_rotate_right(right))
      }
      self.root_index_rotate_left(edge)
    } else {
      edge
    }
    edge = self.root_index_parents[balanced]
  }
}

///|
fn PrefixIndex::root_index_insert(self : PrefixIndex, root : Int) -> Unit {
  let edge = self.root_index_free_head
  if edge < 0 {
    abort("prefix root-index preflight invariant")
  }
  self.root_index_free_head = self.root_index_free_next[edge]
  self.root_index_free_next[edge] = -1
  self.free_stack_touches += 1
  self.root_index_payloads[edge] = root
  self.root_index_left[edge] = -1
  self.root_index_right[edge] = -1
  self.root_index_parents[edge] = -1
  self.root_index_heights[edge] = 1
  self.root_index_slots[root] = edge
  if self.root_index_root < 0 {
    self.root_index_root = edge
    return
  }
  let mut current = self.root_index_root
  for ;; {
    let order = self.compare_root_fixed(root, self.root_index_payloads[current])
    if order == 0 {
      abort("prefix duplicate root-index invariant")
    }
    let next = if order < 0 {
      self.root_index_left[current]
    } else {
      self.root_index_right[current]
    }
    if next >= 0 {
      current = next
    } else {
      self.root_index_parents[edge] = current
      if order < 0 {
        self.root_index_left[current] = edge
      } else {
        self.root_index_right[current] = edge
      }
      self.root_index_rebalance(current)
      return
    }
  }
}

///|
fn PrefixIndex::root_index_remove(self : PrefixIndex, root : Int) -> Unit {
  let mut edge = self.root_index_slots[root]
  if edge < 0 {
    abort("prefix root-index removal invariant")
  }
  if self.root_index_left[edge] >= 0 && self.root_index_right[edge] >= 0 {
    let mut successor = self.root_index_right[edge]
    while self.root_index_left[successor] >= 0 {
      successor = self.root_index_left[successor]
    }
    let successor_root = self.root_index_payloads[successor]
    self.root_index_payloads[edge] = successor_root
    self.root_index_slots[successor_root] = edge
    self.root_index_payloads[successor] = root
    self.root_index_slots[root] = successor
    edge = successor
  }
  let child = if self.root_index_left[edge] >= 0 {
    self.root_index_left[edge]
  } else {
    self.root_index_right[edge]
  }
  let parent = self.root_index_parents[edge]
  self.root_index_replace_parent(edge, child)
  self.root_index_slots[root] = -1
  self.root_index_payloads[edge] = -1
  self.root_index_left[edge] = -1
  self.root_index_right[edge] = -1
  self.root_index_parents[edge] = -1
  self.root_index_heights[edge] = 0
  self.root_index_free_next[edge] = self.root_index_free_head
  self.root_index_free_head = edge
  self.free_stack_touches += 1
  self.root_index_rebalance(parent)
}

///|
fn PrefixIndex::find_request_root(
  self : PrefixIndex,
  key : PrefixRequestKey,
) -> Int raise PrefixIndexError {
  let mut edge = self.root_index_root
  while edge >= 0 {
    self.root_lookup_comparisons += 1
    let root = self.root_index_payloads[edge]
    let order = self.compare_request_root(key, root)
    if order == 0 {
      if !self.request_root_matches(root, key) {
        raise InvariantViolation(RootCount)
      }
      return root
    }
    edge = if order < 0 {
      self.root_index_left[edge]
    } else {
      self.root_index_right[edge]
    }
  }
  -1
}

///|
fn PrefixIndex::take_free_root(self : PrefixIndex) -> Int {
  let root = self.root_free_head
  if root < 0 {
    return -1
  }
  self.root_free_head = self.root_free_next[root]
  self.root_free_next[root] = -1
  self.free_stack_touches += 1
  root
}

///|
fn PrefixIndex::release_root_slot(self : PrefixIndex, root : Int) -> Unit {
  self.root_free_next[root] = self.root_free_head
  self.root_free_head = root
  self.free_stack_touches += 1
}
