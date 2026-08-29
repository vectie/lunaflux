///|
pub fn PrefixTokenBuffer::new(
  capacity : Int,
) -> PrefixTokenBuffer raise PrefixIndexError {
  if !valid_capacity(capacity) {
    raise InvalidLimit(TokenCapacity)
  }
  { tokens: FixedArray::make(capacity, 0), length: 0 }
}

///|
pub fn PrefixTokenBuffer::capacity(self : PrefixTokenBuffer) -> Int {
  self.tokens.length()
}

///|
pub fn PrefixTokenBuffer::length(self : PrefixTokenBuffer) -> Int {
  self.length
}

///|
pub fn PrefixTokenBuffer::clear(self : PrefixTokenBuffer) -> Unit {
  self.length = 0
}

///|
pub fn PrefixTokenBuffer::push_token(
  self : PrefixTokenBuffer,
  token : Int,
) -> Unit raise PrefixIndexError {
  if token < 0 {
    raise InvalidToken(self.length)
  }
  if self.length >= self.tokens.length() {
    raise Exhausted
  }
  self.tokens[self.length] = token
  self.length += 1
}

///|
fn PrefixTokenBuffer::token_at(self : PrefixTokenBuffer, index : Int) -> Int {
  self.tokens[index]
}

///|
pub fn PrefixPublishBuffer::new(
  capacity : Int,
) -> PrefixPublishBuffer raise PrefixIndexError {
  if !valid_capacity(capacity) {
    raise InvalidLimit(PageCapacity)
  }
  let pages = @page_allocator.PageIdStorage::new(capacity) catch {
    _ => raise InvalidLimit(PageCapacity)
  }
  {
    pages,
    planned: FixedArray::make(capacity, false),
    adopted: FixedArray::make(capacity, false),
    length: 0,
  }
}

///|
pub fn PrefixPublishBuffer::capacity(self : PrefixPublishBuffer) -> Int {
  self.adopted.length()
}

///|
pub fn PrefixPublishBuffer::length(self : PrefixPublishBuffer) -> Int {
  self.length
}

///|
pub fn PrefixPublishBuffer::clear(self : PrefixPublishBuffer) -> Unit {
  for index in 0..<self.length {
    try! self.pages.clear(index)
    self.planned[index] = false
    self.adopted[index] = false
  }
  self.length = 0
}

///|
pub fn PrefixPublishBuffer::push_page(
  self : PrefixPublishBuffer,
  page : @page_allocator.PageId,
) -> Unit raise PrefixIndexError {
  if self.length >= self.adopted.length() {
    raise ResultBufferTooSmall
  }
  self.pages.write(self.length, page) catch {
    _ => raise ResultBufferTooSmall
  }
  self.planned[self.length] = false
  self.adopted[self.length] = false
  self.length += 1
}

///|
pub fn PrefixPublishBuffer::page_at(
  self : PrefixPublishBuffer,
  index : Int,
) -> @page_allocator.PageId raise PrefixIndexError {
  if index < 0 || index >= self.length {
    raise ResultIndexOutOfBounds
  }
  self.pages.read(index) catch {
    _ => raise ResultIndexOutOfBounds
  }
}

///|
pub fn PrefixPublishBuffer::was_adopted(
  self : PrefixPublishBuffer,
  index : Int,
) -> Bool raise PrefixIndexError {
  if index < 0 || index >= self.length {
    raise ResultIndexOutOfBounds
  }
  self.adopted[index]
}

///|
pub fn PrefixPublishBuffer::planned_adoption(
  self : PrefixPublishBuffer,
  index : Int,
) -> Bool raise PrefixIndexError {
  if index < 0 || index >= self.length {
    raise ResultIndexOutOfBounds
  }
  self.planned[index]
}
