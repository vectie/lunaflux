///|
pub fn PrefixLookupBuffer::new(
  page_capacity : Int,
) -> PrefixLookupBuffer raise PrefixIndexError {
  if !valid_capacity(page_capacity) {
    raise InvalidLimit(PageCapacity)
  }
  {
    pages: @page_allocator.PageIdStorage::new(page_capacity) catch {
      _ => raise InvalidLimit(PageCapacity)
    },
    page_count: 0,
    matched_tokens: 0,
    entry_index: -1,
    entry_generation: 0,
  }
}

///|
pub fn PrefixLookupBuffer::capacity(self : PrefixLookupBuffer) -> Int {
  self.pages.capacity()
}

///|
pub fn PrefixLookupBuffer::page_count(self : PrefixLookupBuffer) -> Int {
  self.page_count
}

///|
pub fn PrefixLookupBuffer::matched_tokens(self : PrefixLookupBuffer) -> Int {
  self.matched_tokens
}

///|
pub fn PrefixLookupBuffer::entry(self : PrefixLookupBuffer) -> PrefixEntryId? {
  if self.entry_index < 0 {
    None
  } else {
    Some(PrefixEntryId(self.entry_index, self.entry_generation))
  }
}

///|
pub fn PrefixLookupBuffer::has_entry(self : PrefixLookupBuffer) -> Bool {
  self.entry_index >= 0
}

///|
pub fn PrefixLookupBuffer::entry_id(
  self : PrefixLookupBuffer,
) -> PrefixEntryId raise PrefixIndexError {
  if self.entry_index < 0 {
    raise ResultIndexOutOfBounds
  }
  PrefixEntryId(self.entry_index, self.entry_generation)
}

///|
pub fn PrefixLookupBuffer::page_at(
  self : PrefixLookupBuffer,
  index : Int,
) -> @page_allocator.PageId raise PrefixIndexError {
  if index < 0 || index >= self.page_count {
    raise ResultIndexOutOfBounds
  }
  self.pages.read(index) catch {
    _ => raise ResultIndexOutOfBounds
  }
}

///|
pub fn PrefixLookupBuffer::clear(self : PrefixLookupBuffer) -> Unit {
  for index in 0..<self.page_count {
    try! self.pages.clear(index)
  }
  self.page_count = 0
  self.matched_tokens = 0
  self.entry_index = -1
  self.entry_generation = 0
}

///|
pub fn PrefixEvictionBuffer::new(
  page_capacity : Int,
) -> PrefixEvictionBuffer raise PrefixIndexError {
  if !valid_capacity(page_capacity) {
    raise InvalidLimit(PageCapacity)
  }
  {
    pages: @page_allocator.PageIdStorage::new(page_capacity) catch {
      _ => raise InvalidLimit(PageCapacity)
    },
    page_count: 0,
    entry_index: -1,
    entry_generation: 0,
  }
}

///|
pub fn PrefixEvictionBuffer::capacity(self : PrefixEvictionBuffer) -> Int {
  self.pages.capacity()
}

///|
pub fn PrefixEvictionBuffer::page_count(self : PrefixEvictionBuffer) -> Int {
  self.page_count
}

///|
pub fn PrefixEvictionBuffer::entry(
  self : PrefixEvictionBuffer,
) -> PrefixEntryId? {
  if self.entry_index < 0 {
    None
  } else {
    Some(PrefixEntryId(self.entry_index, self.entry_generation))
  }
}

///|
pub fn PrefixEvictionBuffer::page_at(
  self : PrefixEvictionBuffer,
  index : Int,
) -> @page_allocator.PageId raise PrefixIndexError {
  if index < 0 || index >= self.page_count {
    raise ResultIndexOutOfBounds
  }
  self.pages.read(index) catch {
    _ => raise ResultIndexOutOfBounds
  }
}

///|
pub fn PrefixEvictionBuffer::clear(self : PrefixEvictionBuffer) -> Unit {
  for index in 0..<self.page_count {
    try! self.pages.clear(index)
  }
  self.page_count = 0
  self.entry_index = -1
  self.entry_generation = 0
}
