///|
fn page_storage_new(
  capacity : Int32,
  failure : BlockTableLimitRule,
) -> @page_allocator.PageIdStorage raise BlockTableError {
  @page_allocator.PageIdStorage::new(capacity) catch {
    _ => raise InvalidLimit(failure)
  }
}

///|
fn page_storage_read(
  storage : @page_allocator.PageIdStorage,
  index : Int32,
) -> @page_allocator.PageId raise BlockTableError {
  storage.read(index) catch {
    _ => raise InvariantViolation(TableShape)
  }
}

///|
fn page_storage_write(
  storage : @page_allocator.PageIdStorage,
  index : Int32,
  page : @page_allocator.PageId,
) -> Unit raise BlockTableError {
  storage.write(index, page) catch {
    _ => raise InvariantViolation(TableShape)
  }
}

///|
fn page_storage_clear(
  storage : @page_allocator.PageIdStorage,
  index : Int32,
) -> Unit raise BlockTableError {
  storage.clear(index) catch {
    _ => raise InvariantViolation(TableShape)
  }
}

///|
fn page_storage_is_occupied(
  storage : @page_allocator.PageIdStorage,
  index : Int32,
) -> Bool raise BlockTableError {
  storage.is_occupied(index) catch {
    _ => raise InvariantViolation(TableShape)
  }
}

///|
fn page_run_read(
  run : @page_allocator.PageRunBuffer,
  index : Int32,
) -> @page_allocator.PageId raise BlockTableError {
  run.page_at(index) catch {
    _ => raise InvariantViolation(TableShape)
  }
}
