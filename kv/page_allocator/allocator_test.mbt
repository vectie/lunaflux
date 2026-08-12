///|
fn test_limits(
  capacity : Int,
  run : Int,
  active? : Int = 8,
  cached? : Int = 8,
  generation? : Int = 100,
) -> @page_allocator.PageAllocatorLimits {
  try! @page_allocator.PageAllocatorLimits::new(
    page_capacity=capacity,
    max_run_pages=run,
    max_active_references_per_page=active,
    max_cached_references_per_page=cached,
    max_generation=generation,
  )
}

///|
fn snapshot_page(
  allocator : @page_allocator.PageAllocator,
  index : Int,
) -> @page_allocator.PageSnapshot raise @page_allocator.PageAllocatorError {
  allocator.debug_snapshot().page_at(index)
}

///|
test "limits reject unbounded and inconsistent construction" {
  try
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=0,
      max_run_pages=1,
      max_active_references_per_page=1,
      max_cached_references_per_page=1,
    )
  catch {
    error => debug_inspect(error, content="InvalidLimit(PageCapacity)")
  } noraise {
    _ => fail("expected page-capacity rejection")
  }
  try
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=4,
      max_run_pages=5,
      max_active_references_per_page=1,
      max_cached_references_per_page=1,
    )
  catch {
    error => debug_inspect(error, content="InvalidLimit(RunCapacity)")
  } noraise {
    _ => fail("expected run-capacity rejection")
  }
  try
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=4,
      max_run_pages=1,
      max_active_references_per_page=0,
      max_cached_references_per_page=1,
    )
  catch {
    error => debug_inspect(error, content="InvalidLimit(ActiveReferences)")
  } noraise {
    _ => fail("expected active-reference rejection")
  }
  try
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=4,
      max_run_pages=1,
      max_active_references_per_page=1,
      max_cached_references_per_page=0,
    )
  catch {
    error => debug_inspect(error, content="InvalidLimit(CachedReferences)")
  } noraise {
    _ => fail("expected cached-reference rejection")
  }
  try
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=4,
      max_run_pages=1,
      max_active_references_per_page=1,
      max_cached_references_per_page=1,
      max_generation=0,
    )
  catch {
    error => debug_inspect(error, content="InvalidLimit(Generation)")
  } noraise {
    _ => fail("expected generation rejection")
  }
}

///|
test "single allocation is deterministic and exhaustion is typed" {
  let allocator = @page_allocator.PageAllocator::new(test_limits(2, 2))
  let first = allocator.allocate_one()
  let second = allocator.allocate_one()
  assert_eq(first.index(), 0)
  assert_eq(first.generation(), 1)
  assert_eq(second.index(), 1)
  assert_eq(allocator.free_page_count(), 0)
  assert_eq(allocator.allocated_page_count(), 2)
  try allocator.allocate_one() catch {
    error => debug_inspect(error, content="Exhausted")
  } noraise {
    _ => fail("expected allocator exhaustion")
  }
  allocator.debug_check_invariants()
}

///|
test "exact run failure leaves allocator and reusable output unchanged" {
  let allocator = @page_allocator.PageAllocator::new(test_limits(2, 2))
  let output = @page_allocator.PageRunBuffer::new(2)
  allocator.allocate_run(2, output)
  let first = output.page_at(0)
  let second = output.page_at(1)
  assert_eq(output.length(), 2)
  assert_eq(output.capacity(), 2)
  try allocator.allocate_run(1, output) catch {
    error => debug_inspect(error, content="Exhausted")
  } noraise {
    _ => fail("expected exact run exhaustion")
  }
  assert_eq(output.length(), 2)
  assert_eq(output.page_at(0), first)
  assert_eq(output.page_at(1), second)
  assert_eq(allocator.allocated_page_count(), 2)
  allocator.release_active(first)
  allocator.release_active(second)
  allocator.allocate_run(2, output)
  assert_eq(output.capacity(), 2)
  assert_eq(output.page_at(0).generation(), 2)
  assert_eq(output.page_at(1).generation(), 2)
  allocator.debug_check_invariants()
}

///|
test "run validation mutates neither output nor metadata" {
  let allocator = @page_allocator.PageAllocator::new(test_limits(3, 2))
  let output = @page_allocator.PageRunBuffer::new(1)
  try allocator.allocate_run(2, output) catch {
    error => debug_inspect(error, content="RunBufferTooSmall")
  } noraise {
    _ => fail("expected reusable buffer capacity rejection")
  }
  assert_eq(output.length(), 0)
  assert_eq(allocator.free_page_count(), 3)
  try allocator.allocate_run(0, output) catch {
    error => debug_inspect(error, content="InvalidRunLength")
  } noraise {
    _ => fail("expected zero run rejection")
  }
  assert_eq(allocator.free_page_count(), 3)
}

///|
test "active and cached ownership remain independent" {
  let allocator = @page_allocator.PageAllocator::new(test_limits(1, 1))
  let page = allocator.allocate_one()
  assert_eq(allocator.active_reference_count(page), 1)
  assert_eq(allocator.cached_reference_count(page), 0)
  allocator.retain_cached(page)
  assert_eq(allocator.cached_reference_count(page), 1)
  allocator.release_active(page)
  let cached_only = snapshot_page(allocator, 0)
  assert_eq(cached_only.active_references(), 0)
  assert_eq(cached_only.cached_references(), 1)
  assert_eq(cached_only.residency(), Allocated)
  allocator.retain_active(page)
  allocator.release_cached(page)
  let active_only = snapshot_page(allocator, 0)
  assert_eq(active_only.active_references(), 1)
  assert_eq(active_only.cached_references(), 0)
  allocator.release_active(page)
  let free = snapshot_page(allocator, 0)
  assert_eq(free.residency(), Free)
  assert_eq(allocator.free_page_count(), 1)
  allocator.debug_check_invariants()
}

///|
test "reference underflow overflow and double release do not mutate ownership" {
  let allocator = @page_allocator.PageAllocator::new(
    test_limits(1, 1, active=2, cached=1),
  )
  let page = allocator.allocate_one()
  try allocator.release_cached(page) catch {
    error =>
      debug_inspect(error, content="ReferenceUnderflow(PageId(0, 1), Cached)")
  } noraise {
    _ => fail("expected cached underflow")
  }
  allocator.retain_active(page)
  try allocator.retain_active(page) catch {
    error =>
      debug_inspect(error, content="ReferenceOverflow(PageId(0, 1), Active)")
  } noraise {
    _ => fail("expected active overflow")
  }
  allocator.retain_cached(page)
  try allocator.retain_cached(page) catch {
    error =>
      debug_inspect(error, content="ReferenceOverflow(PageId(0, 1), Cached)")
  } noraise {
    _ => fail("expected cached overflow")
  }
  allocator.release_active(page)
  allocator.release_active(page)
  allocator.release_cached(page)
  try allocator.release_active(page) catch {
    error => debug_inspect(error, content="PageNotAllocated(PageId(0, 1))")
  } noraise {
    _ => fail("expected double release rejection")
  }
  assert_eq(allocator.free_page_count(), 1)
  allocator.debug_check_invariants()
}

///|
test "reuse invalidates stale identities and preserves FIFO order" {
  let allocator = @page_allocator.PageAllocator::new(test_limits(3, 1))
  let first = allocator.allocate_one()
  let second = allocator.allocate_one()
  allocator.release_active(first)
  let third = allocator.allocate_one()
  assert_eq(third.index(), 2)
  allocator.release_active(second)
  allocator.release_active(third)
  let reused_first = allocator.allocate_one()
  assert_eq(reused_first.index(), 0)
  assert_eq(reused_first.generation(), 2)
  try allocator.retain_active(first) catch {
    error => debug_inspect(error, content="StalePage(PageId(0, 1))")
  } noraise {
    _ => fail("expected stale identity rejection")
  }
  allocator.debug_check_invariants()
}

///|
test "terminal generations retire instead of wrapping" {
  let allocator = @page_allocator.PageAllocator::new(
    test_limits(1, 1, generation=2),
  )
  let first = allocator.allocate_one()
  allocator.release_active(first)
  let second = allocator.allocate_one()
  assert_eq(second.generation(), 2)
  allocator.release_active(second)
  assert_eq(allocator.free_page_count(), 0)
  assert_eq(allocator.allocated_page_count(), 0)
  assert_eq(allocator.retired_page_count(), 1)
  assert_eq(snapshot_page(allocator, 0).residency(), Retired)
  try allocator.allocate_one() catch {
    error => debug_inspect(error, content="Exhausted")
  } noraise {
    _ => fail("expected retired allocator exhaustion")
  }
  allocator.debug_check_invariants()
}

///|
test "run buffer never exposes a mutable internal array" {
  let allocator = @page_allocator.PageAllocator::new(test_limits(2, 2))
  let output = @page_allocator.PageRunBuffer::new(2)
  allocator.allocate_run(2, output)
  let copied_id = output.page_at(0)
  allocator.release_active(copied_id)
  assert_eq(output.page_at(0), copied_id)
  assert_eq(output.capacity(), 2)
  assert_eq(output.length(), 2)
}
