///|
/// Allocates reusable caller-owned output storage during startup.
pub fn PageRunBuffer::new(
  capacity : Int32,
) -> PageRunBuffer raise PageAllocatorError {
  if capacity <= 0 || capacity > ABSOLUTE_MAX_PAGE_CAPACITY {
    raise InvalidLimit(RunCapacity)
  }
  { pages: FixedArray::make(capacity, invalid_page_id()), length: 0, }
}

///|
pub fn PageRunBuffer::capacity(self : PageRunBuffer) -> Int32 {
  self.pages.length()
}

///|
pub fn PageRunBuffer::length(self : PageRunBuffer) -> Int32 {
  self.length
}

///|
/// Returns one copied opaque identity, never a mutable alias to backing data.
pub fn PageRunBuffer::page_at(
  self : PageRunBuffer,
  index : Int32,
) -> PageId raise PageAllocatorError {
  if index < 0 || index >= self.length {
    raise RunIndexOutOfBounds
  }
  self.pages[index]
}
