///|
/// Allocates one reusable exact-position output during startup.
pub fn BlockTablePositionBuffer::new() -> BlockTablePositionBuffer raise BlockTableError {
  {
    page_storage: page_storage_new(1, PageCellCapacity),
    logical_page_index: 0,
    token_offset: 0,
    is_set: false,
  }
}

///|
pub fn BlockTablePositionBuffer::is_set(
  self : BlockTablePositionBuffer,
) -> Bool {
  self.is_set
}

///|
pub fn BlockTablePositionBuffer::page(
  self : BlockTablePositionBuffer,
) -> @page_allocator.PageId raise BlockTableError {
  if !self.is_set {
    raise PositionUnset
  }
  page_storage_read(self.page_storage, 0)
}

///|
pub fn BlockTablePositionBuffer::logical_page_index(
  self : BlockTablePositionBuffer,
) -> Int raise BlockTableError {
  if !self.is_set {
    raise PositionUnset
  }
  self.logical_page_index
}

///|
pub fn BlockTablePositionBuffer::token_offset(
  self : BlockTablePositionBuffer,
) -> Int raise BlockTableError {
  if !self.is_set {
    raise PositionUnset
  }
  self.token_offset
}

///|
pub fn BlockTablePositionBuffer::clear(
  self : BlockTablePositionBuffer,
) -> Unit raise BlockTableError {
  page_storage_clear(self.page_storage, 0)
  self.logical_page_index = 0
  self.token_offset = 0
  self.is_set = false
}

///|
fn BlockTablePositionBuffer::replace(
  self : BlockTablePositionBuffer,
  page : @page_allocator.PageId,
  logical_page_index : Int,
  token_offset : Int,
) -> Unit raise BlockTableError {
  page_storage_write(self.page_storage, 0, page)
  self.logical_page_index = logical_page_index
  self.token_offset = token_offset
  self.is_set = true
}
