///|
fn table_limits(
  tables : Int,
  pages : Int,
  tokens? : Int = 4,
  generation? : Int = 100,
) -> @block_table.BlockTableLimits {
  try! @block_table.BlockTableLimits::new(
    table_capacity=tables,
    max_pages_per_table=pages,
    tokens_per_page=tokens,
    max_generation=generation,
  )
}

///|
fn page_allocator(capacity : Int) -> @page_allocator.PageAllocator {
  let limits = try! @page_allocator.PageAllocatorLimits::new(
    page_capacity=capacity,
    max_run_pages=capacity,
    max_active_references_per_page=4,
    max_cached_references_per_page=2,
  )
  @page_allocator.PageAllocator::new(limits)
}

///|
test "limits reject impossible dense arenas" {
  try
    @block_table.BlockTableLimits::new(
      table_capacity=0,
      max_pages_per_table=1,
      tokens_per_page=1,
    )
  catch {
    error => debug_inspect(error, content="InvalidLimit(TableCapacity)")
  } noraise {
    _ => fail("expected table-capacity rejection")
  }
  try
    @block_table.BlockTableLimits::new(
      table_capacity=1,
      max_pages_per_table=0,
      tokens_per_page=1,
    )
  catch {
    error => debug_inspect(error, content="InvalidLimit(PagesPerTable)")
  } noraise {
    _ => fail("expected pages-per-table rejection")
  }
  try
    @block_table.BlockTableLimits::new(
      table_capacity=17,
      max_pages_per_table=1_048_576,
      tokens_per_page=1,
    )
  catch {
    error => debug_inspect(error, content="InvalidLimit(PageCellCapacity)")
  } noraise {
    _ => fail("expected dense-cell rejection")
  }
  try
    @block_table.BlockTableLimits::new(
      table_capacity=1,
      max_pages_per_table=1,
      tokens_per_page=0,
    )
  catch {
    error => debug_inspect(error, content="InvalidLimit(TokensPerPage)")
  } noraise {
    _ => fail("expected token-page rejection")
  }
  try
    @block_table.BlockTableLimits::new(
      table_capacity=1,
      max_pages_per_table=1,
      tokens_per_page=1,
      max_generation=0,
    )
  catch {
    error => debug_inspect(error, content="InvalidLimit(Generation)")
  } noraise {
    _ => fail("expected generation rejection")
  }
  try @block_table.BlockTablePageBuffer::new(0) catch {
    error => debug_inspect(error, content="InvalidLimit(PagesPerTable)")
  } noraise {
    _ => fail("expected buffer-capacity rejection")
  }
}

///|
test "allocation and reuse follow deterministic FIFO generations" {
  let arena = @block_table.BlockTableArena::new(table_limits(3, 2))
  let first = arena.allocate()
  let second = arena.allocate()
  let third = arena.allocate()
  assert_eq(first.index(), 0)
  assert_eq(second.index(), 1)
  assert_eq(third.index(), 2)
  try arena.allocate() catch {
    error => debug_inspect(error, content="Exhausted")
  } noraise {
    _ => fail("expected arena exhaustion")
  }
  arena.release(first)
  arena.release(second)
  let reused_first = arena.allocate()
  assert_eq(reused_first.index(), 0)
  assert_eq(reused_first.generation(), 2)
  try arena.page_count(first) catch {
    error => debug_inspect(error, content="StaleTable(BlockTableId(0, 1))")
  } noraise {
    _ => fail("expected stale table rejection")
  }
  arena.debug_check_invariants()
}

///|
test "dense mapping resolves exact logical pages and token offsets" {
  let physical = page_allocator(3)
  let pages = [
    physical.allocate_one(),
    physical.allocate_one(),
    physical.allocate_one(),
  ]
  let arena = @block_table.BlockTableArena::new(table_limits(1, 3, tokens=4))
  let table = arena.allocate()
  arena.append_pages(table, pages)
  let position = @block_table.BlockTablePositionBuffer::new()
  assert_false(position.is_set())
  try position.page() catch {
    error => debug_inspect(error, content="PositionUnset")
  } noraise {
    _ => fail("expected unset-position rejection")
  }
  assert_eq(arena.page_count(table), 3)
  assert_eq(arena.page_at(table, 1), pages[1])
  arena.resolve_token_position(table, 0, position)
  assert_eq(position.page(), pages[0])
  assert_eq(position.logical_page_index(), 0)
  assert_eq(position.token_offset(), 0)
  arena.resolve_token_position(table, 8, position)
  assert_eq(position.page(), pages[2])
  assert_eq(position.logical_page_index(), 2)
  assert_eq(position.token_offset(), 0)
  arena.resolve_token_position(table, 11, position)
  assert_eq(position.page(), pages[2])
  assert_eq(position.token_offset(), 3)
  try arena.resolve_token_position(table, 12, position) catch {
    error =>
      debug_inspect(
        error,
        content="TokenPositionOutOfBounds(BlockTableId(0, 1))",
      )
  } noraise {
    _ => fail("expected token-position rejection")
  }
  assert_eq(position.page(), pages[2])
  assert_eq(position.logical_page_index(), 2)
  assert_eq(position.token_offset(), 3)
  position.clear()
  assert_false(position.is_set())
  try arena.page_at(table, 3) catch {
    error =>
      debug_inspect(
        error,
        content="LogicalPageIndexOutOfBounds(BlockTableId(0, 1))",
      )
  } noraise {
    _ => fail("expected logical-page rejection")
  }
  arena.debug_check_invariants()
}

///|
test "append failures are transactional" {
  let physical = page_allocator(3)
  let pages = [physical.allocate_one(), physical.allocate_one()]
  let extra = [physical.allocate_one()]
  let arena = @block_table.BlockTableArena::new(table_limits(1, 2))
  let table = arena.allocate()
  try arena.append_pages(table, []) catch {
    error => debug_inspect(error, content="InvalidPageCount")
  } noraise {
    _ => fail("expected empty append rejection")
  }
  assert_eq(arena.page_count(table), 0)
  arena.append_pages(table, pages)
  try arena.append_pages(table, extra) catch {
    error => debug_inspect(error, content="TableFull(BlockTableId(0, 1))")
  } noraise {
    _ => fail("expected full-table rejection")
  }
  assert_eq(arena.page_count(table), 2)
  assert_eq(arena.page_at(table, 0), pages[0])
  assert_eq(arena.page_at(table, 1), pages[1])
  try arena.release(table) catch {
    error => debug_inspect(error, content="TableNotEmpty(BlockTableId(0, 1))")
  } noraise {
    _ => fail("expected non-empty release rejection")
  }
  arena.debug_check_invariants()
}

///|
test "allocator runs and single pages append without temporary arrays" {
  let physical = page_allocator(3)
  let run = @page_allocator.PageRunBuffer::new(2)
  physical.allocate_run(2, run)
  let tail = physical.allocate_one()
  let arena = @block_table.BlockTableArena::new(table_limits(1, 3, tokens=4))
  let table = arena.allocate()
  arena.append_run(table, run)
  arena.append_page(table, tail)
  assert_eq(arena.page_count(table), 3)
  assert_eq(arena.page_at(table, 0), run.page_at(0))
  assert_eq(arena.page_at(table, 1), run.page_at(1))
  assert_eq(arena.page_at(table, 2), tail)
  let position = @block_table.BlockTablePositionBuffer::new()
  arena.resolve_token_position(table, 11, position)
  assert_eq(position.page(), tail)
  assert_eq(position.logical_page_index(), 2)
  assert_eq(position.token_offset(), 3)
  let removed = @block_table.BlockTablePageBuffer::new(2)
  arena.truncate(table, 1, removed)
  assert_eq(arena.page_count(table), 1)
  assert_eq(removed.length(), 2)
  assert_eq(removed.page_at(0), run.page_at(1))
  assert_eq(removed.page_at(1), tail)
  try arena.resolve_token_position(table, 4, position) catch {
    error =>
      debug_inspect(
        error,
        content="TokenPositionOutOfBounds(BlockTableId(0, 1))",
      )
  } noraise {
    _ => fail("expected truncated boundary rejection")
  }
  // Logical rollback does not silently mutate physical ownership.
  assert_eq(physical.active_reference_count(removed.page_at(0)), 1)
  assert_eq(physical.active_reference_count(removed.page_at(1)), 1)
  physical.release_active(removed.page_at(0))
  physical.release_active(removed.page_at(1))
  let final_page = @block_table.BlockTablePageBuffer::new(1)
  arena.reset_and_release(table, final_page)
  physical.release_active(final_page.page_at(0))
  assert_eq(physical.free_page_count(), 3)
  arena.debug_check_invariants()
  physical.debug_check_invariants()
}

///|
test "truncate preflight preserves table and reusable output" {
  let physical = page_allocator(4)
  let pages = [
    physical.allocate_one(),
    physical.allocate_one(),
    physical.allocate_one(),
    physical.allocate_one(),
  ]
  let arena = @block_table.BlockTableArena::new(table_limits(2, 3))
  let target = arena.allocate()
  let seed = arena.allocate()
  arena.append_pages(target, pages[0:3])
  arena.append_page(seed, pages[3])
  let output = @block_table.BlockTablePageBuffer::new(1)
  arena.copy_pages(seed, output)
  assert_eq(output.page_at(0), pages[3])
  try arena.truncate(target, 1, output) catch {
    error => debug_inspect(error, content="PageBufferTooSmall")
  } noraise {
    _ => fail("expected truncate buffer rejection")
  }
  assert_eq(arena.page_count(target), 3)
  assert_eq(arena.page_at(target, 1), pages[1])
  assert_eq(arena.page_at(target, 2), pages[2])
  assert_eq(output.length(), 1)
  assert_eq(output.page_at(0), pages[3])
  try arena.truncate(target, 4, output) catch {
    error =>
      debug_inspect(error, content="InvalidTruncateLength(BlockTableId(0, 1))")
  } noraise {
    _ => fail("expected invalid truncate rejection")
  }
  assert_eq(arena.page_count(target), 3)
  assert_eq(output.page_at(0), pages[3])
  arena.debug_check_invariants()
}

///|
test "copy and reset preserve output on preflight failure" {
  let physical = page_allocator(3)
  let pages = [
    physical.allocate_one(),
    physical.allocate_one(),
    physical.allocate_one(),
  ]
  let arena = @block_table.BlockTableArena::new(table_limits(2, 3))
  let first = arena.allocate()
  let second = arena.allocate()
  arena.append_pages(first, pages[0:2])
  arena.append_pages(second, pages[2:3])
  let small = @block_table.BlockTablePageBuffer::new(1)
  arena.copy_pages(second, small)
  assert_eq(small.length(), 1)
  assert_eq(small.page_at(0), pages[2])
  try arena.reset(first, small) catch {
    error => debug_inspect(error, content="PageBufferTooSmall")
  } noraise {
    _ => fail("expected reset preflight rejection")
  }
  assert_eq(arena.page_count(first), 2)
  assert_eq(small.length(), 1)
  assert_eq(small.page_at(0), pages[2])
  let output = @block_table.BlockTablePageBuffer::new(3)
  arena.copy_pages(first, output)
  assert_eq(output.length(), 2)
  assert_eq(output.page_at(0), pages[0])
  assert_eq(output.page_at(1), pages[1])
  assert_eq(arena.page_count(first), 2)
  arena.reset(first, output)
  assert_eq(output.length(), 2)
  assert_eq(arena.page_count(first), 0)
  arena.release(first)
  arena.debug_check_invariants()
}

///|
test "metadata draining never changes physical page ownership" {
  let physical = page_allocator(2)
  let pages = [physical.allocate_one(), physical.allocate_one()]
  let arena = @block_table.BlockTableArena::new(table_limits(1, 2))
  let table = arena.allocate()
  arena.append_pages(table, pages)
  let output = @block_table.BlockTablePageBuffer::new(2)
  arena.reset_and_release(table, output)
  assert_eq(output.length(), 2)
  assert_eq(physical.active_reference_count(pages[0]), 1)
  assert_eq(physical.active_reference_count(pages[1]), 1)
  assert_eq(physical.allocated_page_count(), 2)
  physical.release_active(output.page_at(0))
  physical.release_active(output.page_at(1))
  assert_eq(physical.free_page_count(), 2)
  try arena.page_count(table) catch {
    error =>
      debug_inspect(error, content="TableNotAllocated(BlockTableId(0, 1))")
  } noraise {
    _ => fail("expected released table rejection")
  }
  arena.debug_check_invariants()
  physical.debug_check_invariants()
}

///|
test "terminal table generations retire rather than wrap" {
  let arena = @block_table.BlockTableArena::new(
    table_limits(1, 1, generation=2),
  )
  let first = arena.allocate()
  arena.release(first)
  let second = arena.allocate()
  assert_eq(second.generation(), 2)
  arena.release(second)
  assert_eq(arena.free_table_count(), 0)
  assert_eq(arena.allocated_table_count(), 0)
  assert_eq(arena.retired_table_count(), 1)
  let slot = arena.debug_snapshot().slot_at(0)
  assert_eq(slot.residency(), Retired)
  try arena.allocate() catch {
    error => debug_inspect(error, content="Exhausted")
  } noraise {
    _ => fail("expected retired arena exhaustion")
  }
  arena.debug_check_invariants()
}
