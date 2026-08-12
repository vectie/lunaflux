///|
/// Allocates bounded inline identity storage during startup.
pub fn PageIdStorage::new(
  capacity : Int,
) -> PageIdStorage raise PageAllocatorError {
  if capacity <= 0 || capacity > ABSOLUTE_MAX_PAGE_ID_STORAGE_CAPACITY {
    raise InvalidLimit(StorageCapacity)
  }
  {
    pages: FixedArray::make(capacity, invalid_page_id()),
    occupied: FixedArray::make(capacity, false),
  }
}

///|
pub fn PageIdStorage::capacity(self : PageIdStorage) -> Int {
  self.pages.length()
}

///|
fn PageIdStorage::check_index(
  self : PageIdStorage,
  index : Int,
) -> Unit raise PageAllocatorError {
  if index < 0 || index >= self.pages.length() {
    raise StorageIndexOutOfBounds
  }
}

///|
pub fn PageIdStorage::is_occupied(
  self : PageIdStorage,
  index : Int,
) -> Bool raise PageAllocatorError {
  self.check_index(index)
  self.occupied[index]
}

///|
/// Copies one canonical inline identity into an occupied cell.
///
/// This storage does not retain a physical page reference and cannot validate
/// allocator residency. The caller keeps that ownership and validation duty.
pub fn PageIdStorage::write(
  self : PageIdStorage,
  index : Int,
  page : PageId,
) -> Unit raise PageAllocatorError {
  self.check_index(index)
  self.pages[index] = page
  self.occupied[index] = true
}

///|
/// Returns a copied canonical identity only from an occupied cell.
pub fn PageIdStorage::read(
  self : PageIdStorage,
  index : Int,
) -> PageId raise PageAllocatorError {
  self.check_index(index)
  if !self.occupied[index] {
    raise StorageCellEmpty
  }
  self.pages[index]
}

///|
/// Clears occupancy without exposing the package-private sentinel.
pub fn PageIdStorage::clear(
  self : PageIdStorage,
  index : Int,
) -> Unit raise PageAllocatorError {
  self.check_index(index)
  self.pages[index] = invalid_page_id()
  self.occupied[index] = false
}
