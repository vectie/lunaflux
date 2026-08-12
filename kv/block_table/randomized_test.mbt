///|
fn block_next_random(seed : Int64) -> Int64 {
  (seed * 1103515245L + 12345L) & 2147483647L
}

///|
fn first_used_table(states : Array[Int], start : Int) -> Int {
  for offset in 0..<states.length() {
    let index = (start + offset) % states.length()
    if states[index] == 1 {
      return index
    }
  }
  -1
}

///|
test "deterministic randomized table model preserves FIFO and dense mappings" {
  let table_capacity = 5
  let max_pages = 6
  let physical = page_allocator(16)
  let page_pool : Array[@page_allocator.PageId] = []
  for _ in 0..<16 {
    page_pool.push(physical.allocate_one())
  }
  let arena = @block_table.BlockTableArena::new(
    table_limits(table_capacity, max_pages),
  )
  let output = @block_table.BlockTablePageBuffer::new(max_pages)
  let ids : Array[@block_table.BlockTableId?] = Array::make(
    table_capacity,
    None,
  )
  let generations = Array::make(table_capacity, 0)
  let states = Array::make(table_capacity, 0)
  let model_pages : Array[Array[@page_allocator.PageId]] = []
  for _ in 0..<table_capacity {
    model_pages.push([])
  }
  let free_fifo : Array[Int] = [0, 1, 2, 3, 4]
  let mut seed = 19L
  for _step in 0..<2000 {
    seed = block_next_random(seed)
    let choice = seed.to_int() % 5
    if choice == 0 && free_fifo.length() > 0 {
      let expected = free_fifo.remove(0)
      let table = arena.allocate()
      assert_eq(table.index(), expected)
      generations[expected] += 1
      assert_eq(table.generation(), generations[expected])
      ids[expected] = Some(table)
      states[expected] = 1
    } else {
      seed = block_next_random(seed)
      let index = first_used_table(states, seed.to_int() % table_capacity)
      if index >= 0 {
        guard ids[index] is Some(table) else {
          fail("used model slot must carry an identity")
        }
        let pages = model_pages[index]
        match choice {
          1 =>
            if pages.length() < max_pages {
              seed = block_next_random(seed)
              let page = page_pool[seed.to_int() % page_pool.length()]
              arena.append_pages(table, [page])
              pages.push(page)
            }
          2 => {
            arena.reset(table, output)
            assert_eq(output.length(), pages.length())
            for page_index, page in pages {
              assert_eq(output.page_at(page_index), page)
            }
            pages.clear()
          }
          3 =>
            if pages.length() == 0 {
              arena.release(table)
              states[index] = 0
              ids[index] = None
              free_fifo.push(index)
            }
          _ => {
            arena.reset_and_release(table, output)
            assert_eq(output.length(), pages.length())
            for page_index, page in pages {
              assert_eq(output.page_at(page_index), page)
            }
            pages.clear()
            states[index] = 0
            ids[index] = None
            free_fifo.push(index)
          }
        }
      }
    }
    arena.debug_check_invariants()
    let snapshot = arena.debug_snapshot()
    assert_eq(snapshot.free_table_count(), free_fifo.length())
    assert_eq(
      snapshot.allocated_table_count(),
      table_capacity - free_fifo.length(),
    )
    for index in 0..<table_capacity {
      let slot = snapshot.slot_at(index)
      assert_eq(slot.id().generation(), generations[index])
      assert_eq(slot.page_count(), model_pages[index].length())
      if states[index] == 0 {
        assert_eq(slot.residency(), Free)
      } else {
        assert_eq(slot.residency(), Allocated)
        guard ids[index] is Some(table) else {
          fail("allocated slot lost its identity")
        }
        for page_index, page in model_pages[index] {
          assert_eq(arena.page_at(table, page_index), page)
        }
      }
    }
  }
}
