///|
test "selected-row prefetch window preserves the exact ordered fold and bounded live operands" {
  for fragments in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 63, 64, 65] {
    let gather = projection_selected_row_gather(
      gather_test_dispatch(fragments * 16),
      QueryRowEnds,
    ).unwrap()
    let depth = gather.prefetch_fragments()
    assert_true(depth >= 1 && depth <= 4 && depth <= fragments)
    let window = Array::makei(depth, index => Some(index))
    let consumed : Array[Int] = []
    let loads = Array::make(fragments, 0)
    for slot in 0..<depth {
      loads[slot] += 1
    }
    // The window owns fragment base+slot before that slot is consumed. Each
    // next value is evaluated before consumption, but replaces the slot only
    // after consumption. Thus evaluation can move without moving a reduction.
    for block in 0..<((fragments + depth - 1) / depth) {
      let base = block * depth
      for slot in 0..<depth {
        if base + slot < fragments {
          let next = if base + depth + slot < fragments {
            let future = base + depth + slot
            loads[future] += 1
            Some(future)
          } else {
            None
          }
          let current = window[slot].unwrap()
          assert_eq(current, base + slot)
          consumed.push(current)
          window[slot] = next
        }
      }
    }
    assert_eq(consumed, Array::makei(fragments, i => i))
    assert_eq(loads, Array::make(fragments, 1))
    assert_eq(window, Array::make(depth, None))
  }
}
