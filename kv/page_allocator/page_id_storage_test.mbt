///|
test "page identity storage is bounded and never exposes empty sentinels" {
  try @page_allocator.PageIdStorage::new(0) catch {
    error => debug_inspect(error, content="InvalidLimit(StorageCapacity)")
  } noraise {
    _ => fail("expected zero-capacity rejection")
  }
  try @page_allocator.PageIdStorage::new(16_777_217) catch {
    error => debug_inspect(error, content="InvalidLimit(StorageCapacity)")
  } noraise {
    _ => fail("expected absolute-capacity rejection")
  }
  let storage = @page_allocator.PageIdStorage::new(2)
  assert_eq(storage.capacity(), 2)
  assert_false(storage.is_occupied(0))
  try storage.read(0) catch {
    error => debug_inspect(error, content="StorageCellEmpty")
  } noraise {
    _ => fail("expected empty-cell rejection")
  }
  try storage.read(2) catch {
    error => debug_inspect(error, content="StorageIndexOutOfBounds")
  } noraise {
    _ => fail("expected index rejection")
  }
}

///|
test "page identity storage copies canonical inline identities" {
  let limits = try! @page_allocator.PageAllocatorLimits::new(
    page_capacity=2,
    max_run_pages=1,
    max_active_references_per_page=1,
    max_cached_references_per_page=1,
  )
  let allocator = @page_allocator.PageAllocator::new(limits)
  let first = allocator.allocate_one()
  let second = allocator.allocate_one()
  let storage = @page_allocator.PageIdStorage::new(2)
  storage.write(0, first)
  storage.write(1, second)
  assert_true(storage.is_occupied(0))
  assert_true(storage.is_occupied(1))
  assert_eq(storage.read(0), first)
  assert_eq(storage.read(1), second)
  storage.write(0, second)
  assert_eq(storage.read(0), second)
  storage.clear(0)
  assert_false(storage.is_occupied(0))
  assert_eq(storage.read(1), second)
  try storage.read(0) catch {
    error => debug_inspect(error, content="StorageCellEmpty")
  } noraise {
    _ => fail("expected cleared-cell rejection")
  }
  // Storage is metadata only and did not mutate physical ownership.
  assert_eq(allocator.active_reference_count(first), 1)
  assert_eq(allocator.active_reference_count(second), 1)
}
