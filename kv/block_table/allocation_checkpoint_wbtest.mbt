///|
fn checkpoint_wb_table_limits(generation? : Int = 8) -> BlockTableLimits {
  try! BlockTableLimits::new(
    table_capacity=2,
    max_pages_per_table=1,
    tokens_per_page=1,
    max_generation=generation,
  )
}

///|
test "terminal checkpoint epoch can commit or roll back before exhaustion" {
  let committed = BlockTableArena::new(checkpoint_wb_table_limits())
  committed.allocation_checkpoint_epoch = 0xfffffffffffffffeUL
  let commit_checkpoint = committed.checkpoint()
  let committed_table = committed.allocate()
  committed.commit_checkpoint(commit_checkpoint)
  assert_eq(committed.page_count(committed_table), 0)
  try committed.checkpoint() catch {
    IdentityOverflow(AllocationCheckpointEpochIdentity) => ()
    _ => fail("expected checkpoint identity exhaustion")
  } noraise {
    _ => fail("expected checkpoint to fail")
  }

  let rolled_back = BlockTableArena::new(checkpoint_wb_table_limits())
  rolled_back.allocation_checkpoint_epoch = 0xfffffffffffffffeUL
  let rollback_checkpoint = rolled_back.checkpoint()
  let provisional = rolled_back.allocate()
  rolled_back.rollback_to(rollback_checkpoint)
  assert_eq(rolled_back.free_table_count(), 2)
  try rolled_back.page_count(provisional) catch {
    StaleTable(_) => ()
    _ => fail("expected rolled-back identity rejection")
  } noraise {
    _ => fail("expected page-count lookup to fail")
  }
  try rolled_back.checkpoint() catch {
    IdentityOverflow(AllocationCheckpointEpochIdentity) => ()
    _ => fail("expected checkpoint identity exhaustion")
  } noraise {
    _ => fail("expected checkpoint to fail")
  }
}

///|
test "table generation overflow preflight leaves free FIFO unchanged" {
  let arena = BlockTableArena::new(checkpoint_wb_table_limits(generation=2))
  arena.generations[0] = 2
  try arena.allocate() catch {
    IdentityOverflow(TableGenerationIdentity) => ()
    _ => fail("expected table generation overflow")
  } noraise {
    _ => fail("expected allocation to fail")
  }
  assert_eq(arena.free_table_count(), 2)
  assert_eq(arena.allocated_table_count(), 0)
  assert_eq(arena.free_head, 0)
  assert_eq(arena.free_tail, 1)
  assert_eq(arena.free_next[0], 1)
}

///|
test "checkpoint identity overflow mutates no arena snapshot" {
  let arena = BlockTableArena::new(checkpoint_wb_table_limits())
  arena.allocation_checkpoint_epoch = 0xffffffffffffffffUL
  try arena.checkpoint() catch {
    IdentityOverflow(AllocationCheckpointEpochIdentity) => ()
    _ => fail("expected checkpoint identity overflow")
  } noraise {
    _ => fail("expected checkpoint to fail")
  }
  assert_false(arena.allocation_checkpoint_open)
  assert_eq(arena.allocation_journal_length, 0)
  assert_eq(arena.free_table_count(), 2)
}

///|
test "mapping journal records each appended table exactly once" {
  let physical_limits = @page_allocator.PageAllocatorLimits::new(
    page_capacity=2,
    max_run_pages=2,
    max_active_references_per_page=1,
    max_cached_references_per_page=1,
  )
  let physical = @page_allocator.PageAllocator::new(physical_limits)
  let arena = BlockTableArena::new(
    try! BlockTableLimits::new(
      table_capacity=2,
      max_pages_per_table=2,
      tokens_per_page=1,
    ),
  )
  let table = arena.allocate()
  let checkpoint = arena.checkpoint()
  arena.append_page(table, physical.allocate_one())
  arena.append_page(table, physical.allocate_one())
  assert_eq(arena.mapping_journal_length, 1)
  assert_eq(arena.mapping_journal_indices[0], table.index())
  assert_eq(arena.mapping_journal_baselines[table.index()], 0)
  assert_true(arena.mapping_touched[table.index()])
  arena.detach_suffix_for_checkpoint(checkpoint, table, 0)
  arena.rollback_to(checkpoint)
  assert_eq(arena.mapping_journal_length, 0)
  assert_false(arena.mapping_touched[table.index()])
}
