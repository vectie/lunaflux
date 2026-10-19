///|
test "pool update ownership is physical, unique and independent of row order" {
  let p = @sparse_index_physical_ir.PoolUpdatePlan::new(rows=8, pool=4).unwrap()
  assert_eq(p.workgroups(), 8)
  let writes = [21, 4, 23, -1, 7, 8, 20, 9]
  assert_eq([ for r in 0..<8 => p.owns_update(writes[:], r) ], [
    true, true, false, false, false, true, false, false,
  ])
  assert_true(p.owns_update([23, 21][:], 0))
  assert_false(p.owns_update([23, 21][:], 1))
  assert_false(p.owns_update(writes[:], 8))
  // A later frame writing the same generation still recomputes its pool.
  assert_true(p.owns_update([22][:], 0))
  assert_true(p.owns_update([5][:], 0))
}
