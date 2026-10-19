///|
test "retained sparse domains bound scratch without truncating selected ranks" {
  let plan = @attention_physical_ir.SelectedSparseSchedule::new(
    queries=1,
    heads=64,
    selected_width=2051,
    history=256,
    workers=256,
  ).unwrap()
  assert_eq(plan.workgroups(), 64)
  assert_eq(plan.retained_slots(), 256)
  assert_eq(plan.scratch_bytes(), 131072L)
  assert_eq(plan.region_cells(0), Some((0L, 256L)))
  assert_eq(plan.region_cells(63), Some((32256L, 32512L)))
  assert_eq(plan.region_cells(64), None)
  assert_eq(plan.independent_owner(257), Some(1))
  assert_true(plan.uniqueness(2051) is Some(WindowedBitmap(words=8)))
  assert_true(plan.uniqueness(0) is Some(Pairwise))
  assert_true(plan.uniqueness(2052) is None)
  let wide = @attention_physical_ir.SelectedSparseSchedule::new(
    queries=3,
    heads=4,
    selected_width=65536,
    history=1048576,
    workers=128,
  ).unwrap()
  assert_eq(wide.scratch_bytes(), 6291456L)
  assert_true(wide.uniqueness(65536) is Some(WindowedBitmap(words=128)))
  assert_true(
    @attention_physical_ir.SelectedSparseSchedule::new(
      queries=2147483647,
      heads=2,
      selected_width=1,
      history=1,
      workers=1,
    )
    is None,
  )
}
