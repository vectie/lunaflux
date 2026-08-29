///|
fn PrefixIndex::lookup_rooted(
  self : PrefixIndex,
  root : Int,
  tokens : PrefixTokenBuffer,
  maximum_reusable_tokens : Int,
  output : PrefixLookupBuffer,
) -> Unit raise PrefixIndexError {
  if maximum_reusable_tokens < 0 || maximum_reusable_tokens > tokens.length() {
    raise InvalidTokenCount
  }
  let layout = match self.root_layouts[root] {
    Some(value) => value
    None => raise InvariantViolation(RootCount)
  }
  let page_size = layout.tokens_per_page()
  let usable = maximum_reusable_tokens / page_size * page_size
  if usable == 0 || tokens.length() == 0 {
    output.clear()
    return
  }
  let mut parent = -1
  let mut position = 0
  let mut page_count = 0
  let mut deepest_node = -1
  let mut path_live = true
  while position < tokens.length() && path_live {
    let child = self.find_child(root, parent, tokens.token_at(position))
    if child < 0 {
      break
    }
    let mut cell = self.node_label_heads[child]
    let mut offset = 0
    while offset < self.node_label_lengths[child] && position < tokens.length() {
      if self.token_values[cell] != tokens.token_at(position) {
        path_live = false
        break
      }
      position += 1
      if position <= usable && position % page_size == 0 {
        let page_cell = self.token_page_cells[cell]
        if page_cell < 0 {
          path_live = false
          break
        }
        self.lookup_page_scratch[page_count] = page_cell
        page_count += 1
        deepest_node = child
      }
      offset += 1
      if offset < self.node_label_lengths[child] {
        cell = self.token_next[cell]
      }
    }
    if path_live && offset == self.node_label_lengths[child] {
      parent = child
    }
  }
  if page_count == 0 {
    output.clear()
    return
  }
  if page_count > output.capacity() {
    raise ResultBufferTooSmall
  }
  let entry = self.protecting_entry(deepest_node)
  if entry < 0 {
    raise InvariantViolation(CompressedShape)
  }
  output.clear()
  for index in 0..<page_count {
    let page = self.page_values.read(self.lookup_page_scratch[index]) catch {
      _ => raise InvariantViolation(PageChain)
    }
    output.pages.write(index, page) catch {
      _ => raise InvariantViolation(PageChain)
    }
  }
  output.page_count = page_count
  output.matched_tokens = page_count * page_size
  output.entry_index = entry
  output.entry_generation = self.entry_generations[entry]
}

///|
pub fn PrefixIndex::lookup(
  self : PrefixIndex,
  key : PrefixRequestKey,
  tokens : PrefixTokenBuffer,
  maximum_reusable_tokens~ : Int,
  output : PrefixLookupBuffer,
) -> Unit raise PrefixIndexError {
  if !key.is_live() {
    raise StaleKey
  }
  if key.cache_permission() == Disabled {
    output.clear()
    return
  }
  if tokens.length() > self.limits.max_tokens_per_entry() {
    raise InvalidTokenCount
  }
  let root = self.find_request_root(key)
  if root < 0 {
    output.clear()
    return
  }
  self.lookup_rooted(root, tokens, maximum_reusable_tokens, output)
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
  if self.entry_active_references[index] == 0 {
    self.eviction_heap_remove(index)
  }
  self.entry_active_references[index] += 1
  self.entry_recencies[index] = self.next_recency()
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
  self.entry_active_references[index] -= 1
  if self.entry_active_references[index] == 0 {
    self.eviction_heap_insert(index)
  }
}

///|
pub fn PrefixIndex::active_reference_count(
  self : PrefixIndex,
  entry : PrefixEntryId,
) -> Int raise PrefixIndexError {
  self.entry_active_references[self.checked_entry(entry)]
}

///|
pub fn PrefixIndex::touch(
  self : PrefixIndex,
  entry : PrefixEntryId,
) -> Unit raise PrefixIndexError {
  let index = self.checked_entry(entry)
  self.entry_recencies[index] = self.next_recency()
  if self.entry_active_references[index] == 0 {
    self.eviction_heap_refresh(index)
  }
}
