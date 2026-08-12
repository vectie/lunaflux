///|
fn checkpoint_limits(capacity : Int) -> @page_allocator.PageAllocatorLimits {
  try! @page_allocator.PageAllocatorLimits::new(
    page_capacity=capacity,
    max_run_pages=capacity,
    max_active_references_per_page=2,
    max_cached_references_per_page=2,
  )
}

///|
test "allocation rollback restores exact identities and FIFO order" {
  let allocator = @page_allocator.PageAllocator::new(checkpoint_limits(4))
  let first = allocator.allocate_one()
  allocator.release_active(first)
  // The exact pre-checkpoint FIFO is now [1, 2, 3, 0].
  let checkpoint = allocator.checkpoint()
  let provisional_first = allocator.allocate_one()
  let provisional_run = @page_allocator.PageRunBuffer::new(2)
  allocator.allocate_run(2, provisional_run)
  assert_eq(provisional_first.index(), 1)
  assert_eq(provisional_run.page_at(0).index(), 2)
  assert_eq(provisional_run.page_at(1).index(), 3)
  allocator.rollback_to(checkpoint)
  assert_eq(allocator.free_page_count(), 4)
  assert_eq(allocator.allocated_page_count(), 0)

  let restored = @page_allocator.PageRunBuffer::new(4)
  allocator.allocate_run(4, restored)
  assert_eq(restored.page_at(0), provisional_first)
  assert_eq(restored.page_at(1), provisional_run.page_at(0))
  assert_eq(restored.page_at(2), provisional_run.page_at(1))
  assert_eq(restored.page_at(3).index(), 0)
  assert_eq(restored.page_at(3).generation(), 2)
  allocator.debug_check_invariants()
}

///|
test "allocation rollback restores several runs atomically" {
  let allocator = @page_allocator.PageAllocator::new(checkpoint_limits(6))
  let left = @page_allocator.PageRunBuffer::new(2)
  let right = @page_allocator.PageRunBuffer::new(3)
  let checkpoint = allocator.checkpoint()
  allocator.allocate_run(2, left)
  allocator.allocate_run(3, right)
  let last_page = allocator.allocate_one()
  assert_eq(allocator.free_page_count(), 0)
  allocator.rollback_to(checkpoint)
  assert_eq(allocator.free_page_count(), 6)
  assert_eq(allocator.allocated_page_count(), 0)

  let replay = @page_allocator.PageRunBuffer::new(6)
  allocator.allocate_run(6, replay)
  assert_eq(replay.page_at(0), left.page_at(0))
  assert_eq(replay.page_at(1), left.page_at(1))
  assert_eq(replay.page_at(2), right.page_at(0))
  assert_eq(replay.page_at(3), right.page_at(1))
  assert_eq(replay.page_at(4), right.page_at(2))
  assert_eq(replay.page_at(5), last_page)
  allocator.debug_check_invariants()
}

///|
test "checkpoint commit preserves allocations and consumes its handle" {
  let allocator = @page_allocator.PageAllocator::new(checkpoint_limits(2))
  let checkpoint = allocator.checkpoint()
  let committed = allocator.allocate_one()
  allocator.commit_checkpoint(checkpoint)
  assert_eq(allocator.active_reference_count(committed), 1)
  assert_eq(allocator.free_page_count(), 1)
  try allocator.rollback_to(checkpoint) catch {
    StalePageAllocationCheckpointEpoch => ()
    _ => fail("expected consumed checkpoint rejection")
  } noraise {
    _ => fail("expected rollback to fail")
  }
  allocator.retain_cached(committed)
  allocator.release_active(committed)
  assert_eq(allocator.cached_reference_count(committed), 1)
  allocator.release_cached(committed)
  allocator.debug_check_invariants()
}

///|
test "checkpoint authentication rejects wrong stale duplicate and nested use" {
  let left = @page_allocator.PageAllocator::new(checkpoint_limits(2))
  let right = @page_allocator.PageAllocator::new(checkpoint_limits(2))
  let checkpoint = left.checkpoint()
  try right.rollback_to(checkpoint) catch {
    PageAllocationCheckpointMismatch => ()
    _ => fail("expected checkpoint owner mismatch")
  } noraise {
    _ => fail("expected rollback to fail")
  }
  try left.checkpoint() catch {
    AllocationCheckpointAlreadyOpen => ()
    _ => fail("expected nested checkpoint rejection")
  } noraise {
    _ => fail("expected checkpoint to fail")
  }
  left.rollback_to(checkpoint)
  try left.rollback_to(checkpoint) catch {
    StalePageAllocationCheckpointEpoch => ()
    _ => fail("expected duplicate rollback rejection")
  } noraise {
    _ => fail("expected rollback to fail")
  }
  let stale = left.checkpoint()
  left.commit_checkpoint(stale)
  let current = left.checkpoint()
  try left.commit_checkpoint(stale) catch {
    StalePageAllocationCheckpointEpoch => ()
    _ => fail("expected stale commit rejection")
  } noraise {
    _ => fail("expected commit to fail")
  }
  left.rollback_to(current)
}

///|
test "open checkpoint rejects every reference-count mutation" {
  let allocator = @page_allocator.PageAllocator::new(checkpoint_limits(2))
  let permanent = allocator.allocate_one()
  let checkpoint = allocator.checkpoint()
  let provisional = allocator.allocate_one()
  try allocator.retain_active(permanent) catch {
    AllocationCheckpointReferenceMutationRejected => ()
    _ => fail("expected active retain rejection")
  } noraise {
    _ => fail("expected retain to fail")
  }
  try allocator.retain_cached(provisional) catch {
    AllocationCheckpointReferenceMutationRejected => ()
    _ => fail("expected cached retain rejection")
  } noraise {
    _ => fail("expected retain to fail")
  }
  try allocator.release_active(provisional) catch {
    AllocationCheckpointReferenceMutationRejected => ()
    _ => fail("expected active release rejection")
  } noraise {
    _ => fail("expected release to fail")
  }
  try allocator.release_cached(permanent) catch {
    AllocationCheckpointReferenceMutationRejected => ()
    _ => fail("expected cached release rejection")
  } noraise {
    _ => fail("expected release to fail")
  }
  assert_eq(allocator.active_reference_count(permanent), 1)
  assert_eq(allocator.active_reference_count(provisional), 1)
  allocator.rollback_to(checkpoint)
  assert_eq(allocator.active_reference_count(permanent), 1)
  allocator.release_active(permanent)
  allocator.debug_check_invariants()
}
