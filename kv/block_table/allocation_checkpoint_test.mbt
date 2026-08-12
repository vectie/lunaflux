///|
test "table allocation rollback restores exact identities and FIFO order" {
  let arena = @block_table.BlockTableArena::new(table_limits(4, 1))
  let first = arena.allocate()
  arena.release(first)
  // The exact pre-checkpoint FIFO is [1, 2, 3, 0].
  let checkpoint = arena.checkpoint()
  let second = arena.allocate()
  let third = arena.allocate()
  let fourth = arena.allocate()
  assert_eq(second.index(), 1)
  assert_eq(third.index(), 2)
  assert_eq(fourth.index(), 3)
  arena.rollback_to(checkpoint)
  assert_eq(arena.free_table_count(), 4)
  assert_eq(arena.allocated_table_count(), 0)

  let replay_second = arena.allocate()
  let replay_third = arena.allocate()
  let replay_fourth = arena.allocate()
  let replay_first = arena.allocate()
  assert_eq(replay_second, second)
  assert_eq(replay_third, third)
  assert_eq(replay_fourth, fourth)
  assert_eq(replay_first.index(), 0)
  assert_eq(replay_first.generation(), 2)
  arena.debug_check_invariants()
}

///|
test "checkpoint appends detach from existing and provisional tables" {
  let physical = page_allocator(3)
  let permanent_page = physical.allocate_one()
  let existing_suffix = physical.allocate_one()
  let provisional_page = physical.allocate_one()
  let arena = @block_table.BlockTableArena::new(table_limits(2, 3))
  let existing = arena.allocate()
  arena.append_page(existing, permanent_page)
  let checkpoint = arena.checkpoint()
  let provisional = arena.allocate()
  arena.append_page(existing, existing_suffix)
  arena.append_page(provisional, provisional_page)
  assert_eq(arena.page_count(existing), 2)
  assert_eq(arena.page_count(provisional), 1)

  arena.detach_suffix_for_checkpoint(checkpoint, existing, 1)
  arena.detach_suffix_for_checkpoint(checkpoint, provisional, 0)
  assert_eq(arena.page_count(existing), 1)
  assert_eq(arena.page_at(existing, 0), permanent_page)
  assert_eq(arena.page_count(provisional), 0)
  arena.rollback_to(checkpoint)
  assert_eq(arena.page_count(existing), 1)
  assert_eq(arena.page_at(existing, 0), permanent_page)
  let replay = arena.allocate()
  assert_eq(replay, provisional)
  arena.debug_check_invariants()
}

///|
test "rollback rejects nonempty provisional and undetached existing suffixes" {
  let physical = page_allocator(2)
  let pages = [physical.allocate_one(), physical.allocate_one()]
  let arena = @block_table.BlockTableArena::new(table_limits(2, 2))
  let existing = arena.allocate()
  let checkpoint = arena.checkpoint()
  let provisional = arena.allocate()
  arena.append_page(existing, pages[0])
  arena.append_page(provisional, pages[1])
  try arena.rollback_to(checkpoint) catch {
    BlockTableAllocationCheckpointMismatch => ()
    _ => fail("expected undetached mapping rejection")
  } noraise {
    _ => fail("expected rollback to fail")
  }
  assert_eq(arena.page_count(existing), 1)
  assert_eq(arena.page_count(provisional), 1)
  arena.detach_suffix_for_checkpoint(checkpoint, existing, 0)
  try arena.rollback_to(checkpoint) catch {
    BlockTableAllocationCheckpointMismatch => ()
    _ => fail("expected nonempty provisional rejection")
  } noraise {
    _ => fail("expected rollback to fail")
  }
  arena.detach_suffix_for_checkpoint(checkpoint, provisional, 0)
  arena.rollback_to(checkpoint)
  assert_eq(arena.page_count(existing), 0)
  arena.debug_check_invariants()
}

///|
test "checkpoint commit preserves allocated tables and appended mappings" {
  let physical = page_allocator(1)
  let page = physical.allocate_one()
  let arena = @block_table.BlockTableArena::new(table_limits(1, 1))
  let checkpoint = arena.checkpoint()
  let table = arena.allocate()
  arena.append_page(table, page)
  arena.commit_checkpoint(checkpoint)
  assert_eq(arena.page_count(table), 1)
  assert_eq(arena.page_at(table, 0), page)
  try arena.rollback_to(checkpoint) catch {
    StaleBlockTableAllocationCheckpointEpoch => ()
    _ => fail("expected consumed checkpoint rejection")
  } noraise {
    _ => fail("expected rollback to fail")
  }
  let drained = @block_table.BlockTablePageBuffer::new(1)
  arena.reset_and_release(table, drained)
  assert_eq(drained.page_at(0), page)
  arena.debug_check_invariants()
}

///|
test "checkpoint authenticates owner epoch nesting and detach baseline" {
  let physical = page_allocator(1)
  let page = physical.allocate_one()
  let left = @block_table.BlockTableArena::new(table_limits(1, 1))
  let right = @block_table.BlockTableArena::new(table_limits(1, 1))
  let table = left.allocate()
  let checkpoint = left.checkpoint()
  try right.rollback_to(checkpoint) catch {
    BlockTableAllocationCheckpointMismatch => ()
    _ => fail("expected owner mismatch")
  } noraise {
    _ => fail("expected rollback to fail")
  }
  try left.checkpoint() catch {
    AllocationCheckpointAlreadyOpen => ()
    _ => fail("expected nested checkpoint rejection")
  } noraise {
    _ => fail("expected checkpoint to fail")
  }
  left.append_page(table, page)
  try left.detach_suffix_for_checkpoint(checkpoint, table, 1) catch {
    InvalidCheckpointDetach(_) => ()
    _ => fail("expected baseline mismatch")
  } noraise {
    _ => fail("expected detach to fail")
  }
  left.detach_suffix_for_checkpoint(checkpoint, table, 0)
  left.rollback_to(checkpoint)
  try left.rollback_to(checkpoint) catch {
    StaleBlockTableAllocationCheckpointEpoch => ()
    _ => fail("expected duplicate rollback rejection")
  } noraise {
    _ => fail("expected rollback to fail")
  }
  let stale = left.checkpoint()
  left.commit_checkpoint(stale)
  let current = left.checkpoint()
  try left.commit_checkpoint(stale) catch {
    StaleBlockTableAllocationCheckpointEpoch => ()
    _ => fail("expected stale checkpoint rejection")
  } noraise {
    _ => fail("expected commit to fail")
  }
  left.rollback_to(current)
}

///|
test "checkpoint rejects lifecycle and destructive mapping mutations" {
  let physical = page_allocator(1)
  let page = physical.allocate_one()
  let arena = @block_table.BlockTableArena::new(table_limits(1, 1))
  let table = arena.allocate()
  let checkpoint = arena.checkpoint()
  arena.append_page(table, page)
  let output = @block_table.BlockTablePageBuffer::new(1)
  try arena.truncate(table, 0, output) catch {
    AllocationCheckpointMutationRejected => ()
    _ => fail("expected truncate rejection")
  } noraise {
    _ => fail("expected truncate to fail")
  }
  try arena.reset(table, output) catch {
    AllocationCheckpointMutationRejected => ()
    _ => fail("expected reset rejection")
  } noraise {
    _ => fail("expected reset to fail")
  }
  try arena.reset_and_release(table, output) catch {
    AllocationCheckpointMutationRejected => ()
    _ => fail("expected reset-and-release rejection")
  } noraise {
    _ => fail("expected reset-and-release to fail")
  }
  try arena.release(table) catch {
    AllocationCheckpointMutationRejected => ()
    _ => fail("expected release rejection")
  } noraise {
    _ => fail("expected release to fail")
  }
  arena.detach_suffix_for_checkpoint(checkpoint, table, 0)
  arena.rollback_to(checkpoint)
  assert_eq(arena.page_count(table), 0)
}
