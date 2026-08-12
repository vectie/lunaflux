///|
fn checkpoint_wb_limits(generation? : Int = 8) -> PageAllocatorLimits {
  try! PageAllocatorLimits::new(
    page_capacity=2,
    max_run_pages=2,
    max_active_references_per_page=1,
    max_cached_references_per_page=1,
    max_generation=generation,
  )
}

///|
test "checkpoint identity overflow is checked before snapshot mutation" {
  let allocator = PageAllocator::new(checkpoint_wb_limits())
  allocator.allocation_checkpoint_epoch = 0xffffffffffffffffUL
  try allocator.checkpoint() catch {
    IdentityOverflow(AllocationCheckpointEpochIdentity) => ()
    _ => fail("expected checkpoint identity overflow")
  } noraise {
    _ => fail("expected checkpoint to fail")
  }
  assert_false(allocator.allocation_checkpoint_open)
  assert_eq(allocator.allocation_journal_length, 0)
  assert_eq(allocator.free_page_count(), 2)
}

///|
test "terminal checkpoint epoch can still be committed or rolled back" {
  let committed = PageAllocator::new(checkpoint_wb_limits())
  committed.allocation_checkpoint_epoch = 0xfffffffffffffffeUL
  let commit_checkpoint = committed.checkpoint()
  let committed_page = committed.allocate_one()
  committed.commit_checkpoint(commit_checkpoint)
  assert_eq(committed.active_reference_count(committed_page), 1)
  try committed.checkpoint() catch {
    IdentityOverflow(AllocationCheckpointEpochIdentity) => ()
    _ => fail("expected exhausted checkpoint identity")
  } noraise {
    _ => fail("expected checkpoint to fail")
  }

  let rolled_back = PageAllocator::new(checkpoint_wb_limits())
  rolled_back.allocation_checkpoint_epoch = 0xfffffffffffffffeUL
  let rollback_checkpoint = rolled_back.checkpoint()
  let provisional = rolled_back.allocate_one()
  rolled_back.rollback_to(rollback_checkpoint)
  assert_eq(rolled_back.free_page_count(), 2)
  try rolled_back.active_reference_count(provisional) catch {
    StalePage(_) => ()
    _ => fail("expected rolled-back identity rejection")
  } noraise {
    _ => fail("expected identity lookup to fail")
  }
  try rolled_back.checkpoint() catch {
    IdentityOverflow(AllocationCheckpointEpochIdentity) => ()
    _ => fail("expected exhausted checkpoint identity")
  } noraise {
    _ => fail("expected checkpoint to fail")
  }
}

///|
test "allocation generation overflow preflight mutates no allocator or output" {
  let allocator = PageAllocator::new(checkpoint_wb_limits(generation=2))
  allocator.generations[0] = 2
  let output = PageRunBuffer::new(2)
  try allocator.allocate_run(2, output) catch {
    IdentityOverflow(PageGenerationIdentity) => ()
    _ => fail("expected page generation overflow")
  } noraise {
    _ => fail("expected allocation to fail")
  }
  assert_eq(output.length(), 0)
  assert_eq(allocator.free_page_count(), 2)
  assert_eq(allocator.allocated_page_count(), 0)
  assert_eq(allocator.free_head, 0)
  assert_eq(allocator.free_tail, 1)
  assert_eq(allocator.generations[0], 2)
  assert_eq(allocator.generations[1], 0)
}

///|
test "rollback rejects mutated provisional ownership before any restoration" {
  let allocator = PageAllocator::new(checkpoint_wb_limits())
  let checkpoint = allocator.checkpoint()
  let page = allocator.allocate_one()
  allocator.active_references[page.index()] = 2
  try allocator.rollback_to(checkpoint) catch {
    PageAllocationCheckpointMismatch => ()
    _ => fail("expected provisional ownership mismatch")
  } noraise {
    _ => fail("expected rollback to fail")
  }
  assert_true(allocator.allocation_checkpoint_open)
  assert_eq(allocator.free_page_count(), 1)
  assert_eq(allocator.allocated_page_count(), 1)
  assert_eq(allocator.generations[page.index()], page.generation())
  assert_eq(allocator.active_references[page.index()], 2)
}
