///|
fn PrefixIndex::free_entry(self : PrefixIndex) -> Int {
  self.entry_free_head
}

///|
fn PrefixIndex::take_free_entry(self : PrefixIndex) -> Int {
  let entry = self.entry_free_head
  if entry < 0 {
    return -1
  }
  self.entry_free_head = self.entry_free_next[entry]
  self.entry_free_next[entry] = -1
  self.free_stack_touches += 1
  entry
}

///|
fn PrefixIndex::release_entry_slot(self : PrefixIndex, entry : Int) -> Unit {
  self.entry_free_next[entry] = self.entry_free_head
  self.entry_free_head = entry
  self.free_stack_touches += 1
}

///|
fn PrefixIndex::take_free_token(self : PrefixIndex) -> Int {
  let cell = self.token_free_head
  if cell < 0 {
    return -1
  }
  self.token_free_head = self.token_free_next[cell]
  self.token_free_next[cell] = -1
  self.free_stack_touches += 1
  cell
}

///|
fn PrefixIndex::release_token_cell(self : PrefixIndex, cell : Int) -> Unit {
  self.token_free_next[cell] = self.token_free_head
  self.token_free_head = cell
  self.free_stack_touches += 1
}

///|
fn PrefixIndex::take_free_page(self : PrefixIndex) -> Int {
  let cell = self.page_free_head
  if cell < 0 {
    return -1
  }
  self.page_free_head = self.page_free_next[cell]
  self.page_free_next[cell] = -1
  self.free_stack_touches += 1
  cell
}

///|
fn PrefixIndex::release_page_cell(self : PrefixIndex, cell : Int) -> Unit {
  self.page_free_next[cell] = self.page_free_head
  self.page_free_head = cell
  self.free_stack_touches += 1
}
