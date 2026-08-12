///|
fn next_random(seed : Int64) -> Int64 {
  (seed * 1103515245L + 12345L) & 2147483647L
}

///|
fn first_allocated(state : Array[Int], start : Int) -> Int {
  let capacity = state.length()
  for offset in 0..<capacity {
    let index = (start + offset) % capacity
    if state[index] == 1 {
      return index
    }
  }
  -1
}

///|
test "deterministic randomized model preserves page ownership and FIFO" {
  let capacity = 8
  let limits = try! @page_allocator.PageAllocatorLimits::new(
    page_capacity=capacity,
    max_run_pages=4,
    max_active_references_per_page=4,
    max_cached_references_per_page=3,
    max_generation=10000,
  )
  let allocator = @page_allocator.PageAllocator::new(limits)
  let initial = allocator.debug_snapshot()
  let ids = Array::make(capacity, initial.page_at(0).id())
  let generations = Array::make(capacity, 0)
  let state = Array::make(capacity, 0)
  let active = Array::make(capacity, 0)
  let cached = Array::make(capacity, 0)
  let free_fifo : Array[Int] = [0, 1, 2, 3, 4, 5, 6, 7]
  let mut seed = 7L
  for _step in 0..<2000 {
    seed = next_random(seed)
    let choice = seed.to_int() % 5
    if choice == 0 && free_fifo.length() > 0 {
      let expected_index = free_fifo.remove(0)
      let page = allocator.allocate_one()
      assert_eq(page.index(), expected_index)
      generations[expected_index] += 1
      assert_eq(page.generation(), generations[expected_index])
      ids[expected_index] = page
      state[expected_index] = 1
      active[expected_index] = 1
      cached[expected_index] = 0
    } else {
      seed = next_random(seed)
      let index = first_allocated(state, seed.to_int() % capacity)
      if index >= 0 {
        let page = ids[index]
        match choice {
          1 =>
            if active[index] < 4 {
              allocator.retain_active(page)
              active[index] += 1
            }
          2 =>
            if cached[index] < 3 {
              allocator.retain_cached(page)
              cached[index] += 1
            }
          3 =>
            if active[index] > 0 {
              allocator.release_active(page)
              active[index] -= 1
            }
          _ =>
            if cached[index] > 0 {
              allocator.release_cached(page)
              cached[index] -= 1
            }
        }
        if active[index] == 0 && cached[index] == 0 {
          state[index] = 0
          free_fifo.push(index)
        }
      }
    }
    allocator.debug_check_invariants()
    let snapshot = allocator.debug_snapshot()
    assert_eq(snapshot.free_page_count(), free_fifo.length())
    assert_eq(snapshot.allocated_page_count(), capacity - free_fifo.length())
    for index in 0..<capacity {
      let page = snapshot.page_at(index)
      assert_eq(page.id().generation(), generations[index])
      assert_eq(page.active_references(), active[index])
      assert_eq(page.cached_references(), cached[index])
      if state[index] == 0 {
        assert_eq(page.residency(), Free)
      } else {
        assert_eq(page.residency(), Allocated)
      }
    }
  }
}
