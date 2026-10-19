///|
test "query reuse owns each sequence without sharing sparse membership" {
  let plan = @attention_physical_ir.SelectedQueryReuse::new(
    queries=65,
    heads=32,
    history=8192,
    query_dimension=256,
    value_dimension=256,
    rows=16,
    shared_budget=49152L,
  ).unwrap()
  assert_eq(plan.workgroups(), 160)
  assert_eq(plan.cohort_rows(), 16)
  assert_eq(plan.dispatch_workgroups(), 2080)
  assert_eq(plan.scratch_bytes(), 68157440L)
  assert_true(plan.owns_sequence([2, 1, 2, -1, 1, 3], 0))
  assert_true(plan.owns_sequence([2, 1, 2, -1, 1, 3], 1))
  assert_false(plan.owns_sequence([2, 1, 2, -1, 1, 3], 2))
  assert_false(plan.owns_sequence([2, 1, 2, -1, 1, 3], 3))
  assert_true(plan.owns_sequence([2, 1, 2, -1, 1, 3], 5))
  assert_true(
    @attention_physical_ir.SelectedQueryReuse::new(
      queries=64,
      heads=32,
      history=1048576,
      query_dimension=256,
      value_dimension=256,
      rows=16,
      shared_budget=49152L,
    )
    is None,
  )
}
