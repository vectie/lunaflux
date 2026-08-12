///|
test "block-table identity storage is bounded and hides empty sentinels" {
  try @block_table.BlockTableIdStorage::new(0) catch {
    error => debug_inspect(error, content="InvalidLimit(StorageCapacity)")
  } noraise {
    _ => fail("expected zero-capacity rejection")
  }
  try @block_table.BlockTableIdStorage::new(16_777_217) catch {
    error => debug_inspect(error, content="InvalidLimit(StorageCapacity)")
  } noraise {
    _ => fail("expected absolute-capacity rejection")
  }
  let storage = @block_table.BlockTableIdStorage::new(2)
  assert_eq(storage.capacity(), 2)
  assert_false(storage.is_occupied(0))
  try storage.read(0) catch {
    error => debug_inspect(error, content="StorageCellEmpty")
  } noraise {
    _ => fail("expected empty-cell rejection")
  }
  try storage.is_occupied(-1) catch {
    error => debug_inspect(error, content="StorageIndexOutOfBounds")
  } noraise {
    _ => fail("expected negative-index rejection")
  }
  try storage.read(2) catch {
    error => debug_inspect(error, content="StorageIndexOutOfBounds")
  } noraise {
    _ => fail("expected read index rejection")
  }
  let fixture = table_fixture_id()
  try storage.write(2, fixture) catch {
    error => debug_inspect(error, content="StorageIndexOutOfBounds")
  } noraise {
    _ => fail("expected write index rejection")
  }
  try storage.clear(2) catch {
    error => debug_inspect(error, content="StorageIndexOutOfBounds")
  } noraise {
    _ => fail("expected clear index rejection")
  }
}

///|
fn table_fixture_id() -> @block_table.BlockTableId raise {
  let limits = @block_table.BlockTableLimits::new(
    table_capacity=1,
    max_pages_per_table=1,
    tokens_per_page=1,
  )
  @block_table.BlockTableArena::new(limits).allocate()
}

///|
test "block-table identity storage copies identities and clears independently" {
  let limits = @block_table.BlockTableLimits::new(
    table_capacity=1,
    max_pages_per_table=1,
    tokens_per_page=1,
  )
  let arena = @block_table.BlockTableArena::new(limits)
  let first = arena.allocate()
  let storage = @block_table.BlockTableIdStorage::new(2)
  storage.write(0, first)
  storage.write(1, first)
  assert_true(storage.is_occupied(0))
  assert_true(storage.is_occupied(1))
  assert_eq(storage.read(0), first)
  assert_eq(storage.read(1), first)

  arena.release(first)
  let second = arena.allocate()
  assert_eq(second.index(), first.index())
  assert_true(second.generation() > first.generation())
  // Recycling the source arena slot cannot mutate the copied old identity.
  assert_eq(storage.read(0), first)
  storage.write(1, second)
  assert_eq(storage.read(0), first)
  assert_eq(storage.read(1), second)

  storage.clear(0)
  assert_false(storage.is_occupied(0))
  assert_eq(storage.read(1), second)
  try storage.read(0) catch {
    error => debug_inspect(error, content="StorageCellEmpty")
  } noraise {
    _ => fail("expected cleared-cell rejection")
  }
  // Storage is metadata only and did not change source arena ownership.
  assert_eq(arena.page_count(second), 0)
}
