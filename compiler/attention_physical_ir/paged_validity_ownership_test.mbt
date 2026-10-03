///|
test "replicated page validity covers malformed pages and inactive tails" {
  let plan = PagedValidityOwnership::new(
    tile_rows=32,
    page_rows=8,
    start_alignment=32,
    owners=32,
  ).unwrap()
  assert_eq(plan.page_count(), 4)
  assert_eq(plan.valid_tile(0, [false, false, false, false]), Some(true))
  assert_eq(plan.valid_tile(1, [true, false, false, false]), Some(true))
  assert_eq(plan.valid_tile(8, [true, false, false, false]), Some(true))
  assert_eq(plan.valid_tile(9, [true, false, false, false]), Some(false))
  assert_eq(plan.valid_tile(32, [true, true, true, false]), Some(false))
  assert_eq(plan.valid_tile(32, [true, true, true, true]), Some(true))
  assert_eq(plan.valid_tile(-1, [true, true, true, true]), None)
  assert_eq(plan.valid_tile(33, [true, true, true, true]), None)
  assert_eq(plan.valid_tile(32, [true]), None)
}

///|
test "replicated validity rejects incomplete or unaligned ownership" {
  assert_eq(
    PagedValidityOwnership::new(
      tile_rows=32,
      page_rows=8,
      start_alignment=1,
      owners=32,
    ),
    None,
  )
  assert_eq(
    PagedValidityOwnership::new(
      tile_rows=32,
      page_rows=8,
      start_alignment=32,
      owners=3,
    ),
    None,
  )
}

///|
test "operand validity snapshots survive future key acquisition" {
  let plan = PagedValidityOwnership::new(
    tile_rows=32,
    page_rows=8,
    start_alignment=32,
    owners=32,
  ).unwrap()
  // Every subgroup sees the same immutable page facts, but different operand
  // slots may hold different epochs while K runs ahead of V.
  let old_value = plan.valid_tile(32, [true, true, true, true]).unwrap()
  let future_key = plan.valid_tile(32, [true, false, true, true]).unwrap()
  assert_true(old_value)
  assert_false(future_key)
  for _warp in 0..<8 {
    assert_eq(plan.valid_tile(32, [true, true, true, true]), Some(old_value))
    assert_eq(plan.valid_tile(32, [true, false, true, true]), Some(future_key))
  }
  for old_valid in [false, true] {
    let value_0 = plan.valid_tile(32, [old_valid, true, true, true]).unwrap()
    for future_valid in [false, true] {
      let key_0 = plan.valid_tile(32, [future_valid, true, true, true]).unwrap()
      assert_eq(key_0, future_valid)
      assert_eq(value_0, old_valid)
    }
  }
}
