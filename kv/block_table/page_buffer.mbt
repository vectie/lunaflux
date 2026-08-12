///|
/// Allocates fixed caller-owned storage during startup.
pub fn BlockTablePageBuffer::new(
  capacity : Int,
) -> BlockTablePageBuffer raise BlockTableError {
  if capacity <= 0 || capacity > ABSOLUTE_MAX_PAGES_PER_TABLE {
    raise InvalidLimit(PagesPerTable)
  }
  { pages: page_storage_new(capacity, PagesPerTable), length: 0 }
}

///|
pub fn BlockTablePageBuffer::capacity(self : BlockTablePageBuffer) -> Int {
  self.pages.capacity()
}

///|
pub fn BlockTablePageBuffer::length(self : BlockTablePageBuffer) -> Int {
  self.length
}

///|
/// Returns one copied opaque identity and never exposes mutable backing storage.
pub fn BlockTablePageBuffer::page_at(
  self : BlockTablePageBuffer,
  index : Int,
) -> @page_allocator.PageId raise BlockTableError {
  if index < 0 || index >= self.length {
    raise PageBufferIndexOutOfBounds
  }
  page_storage_read(self.pages, index)
}

///|
fn BlockTablePageBuffer::replace_from_table(
  self : BlockTablePageBuffer,
  arena : BlockTableArena,
  table_index : Int,
  page_count : Int,
) -> Unit raise BlockTableError {
  self.replace_from_range(
    arena,
    table_index * arena.max_pages_per_table,
    page_count,
  )
}

///|
fn BlockTablePageBuffer::replace_from_range(
  self : BlockTablePageBuffer,
  arena : BlockTableArena,
  source_base : Int,
  page_count : Int,
) -> Unit raise BlockTableError {
  for offset in 0..<page_count {
    page_storage_write(
      self.pages,
      offset,
      page_storage_read(arena.pages, source_base + offset),
    )
  }
  for offset in page_count..<self.length {
    page_storage_clear(self.pages, offset)
  }
  self.length = page_count
}
