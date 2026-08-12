///|
pub fn PrefixLookupBuffer::new(
  page_capacity : Int,
) -> PrefixLookupBuffer raise PrefixIndexError {
  if !valid_capacity(page_capacity) {
    raise InvalidLimit(PageCapacity)
  }
  {
    pages: FixedArray::make(page_capacity, None),
    page_count: 0,
    matched_tokens: 0,
    entry: None,
  }
}

///|
pub fn PrefixLookupBuffer::capacity(self : PrefixLookupBuffer) -> Int {
  self.pages.length()
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
  self.entry
}

///|
pub fn PrefixLookupBuffer::page_at(
  self : PrefixLookupBuffer,
  index : Int,
) -> @page_allocator.PageId raise PrefixIndexError {
  if index < 0 || index >= self.page_count {
    raise ResultIndexOutOfBounds
  }
  match self.pages[index] {
    Some(page) => page
    None => raise ResultIndexOutOfBounds
  }
}

///|
pub fn PrefixLookupBuffer::clear(self : PrefixLookupBuffer) -> Unit {
  self.page_count = 0
  self.matched_tokens = 0
  self.entry = None
}

///|
pub fn PrefixEvictionBuffer::new(
  page_capacity : Int,
) -> PrefixEvictionBuffer raise PrefixIndexError {
  if !valid_capacity(page_capacity) {
    raise InvalidLimit(PageCapacity)
  }
  { pages: FixedArray::make(page_capacity, None), page_count: 0, entry: None }
}

///|
pub fn PrefixEvictionBuffer::capacity(self : PrefixEvictionBuffer) -> Int {
  self.pages.length()
}

///|
pub fn PrefixEvictionBuffer::page_count(self : PrefixEvictionBuffer) -> Int {
  self.page_count
}

///|
pub fn PrefixEvictionBuffer::entry(
  self : PrefixEvictionBuffer,
) -> PrefixEntryId? {
  self.entry
}

///|
pub fn PrefixEvictionBuffer::page_at(
  self : PrefixEvictionBuffer,
  index : Int,
) -> @page_allocator.PageId raise PrefixIndexError {
  if index < 0 || index >= self.page_count {
    raise ResultIndexOutOfBounds
  }
  match self.pages[index] {
    Some(page) => page
    None => raise ResultIndexOutOfBounds
  }
}

///|
pub fn PrefixEvictionBuffer::clear(self : PrefixEvictionBuffer) -> Unit {
  self.page_count = 0
  self.entry = None
}
