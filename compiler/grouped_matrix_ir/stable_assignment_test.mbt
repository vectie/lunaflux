///|
test "stable assignment carries integer prefix storage, not numeric arithmetic" {
  for (rows, selected) in [(1, 1), (17, 2), (64, 8)] {
    let p = @grouped_matrix_ir.ExpertGroupingPlan::new(
      maximum_rows=rows,
      selected_experts=selected,
    ).stable_assignment()
    assert_eq(p.capacity(), rows * selected)
    assert_eq(p.scratch_elements(), rows * selected * 3)
    assert_eq(p.rows_per_group(), 16)
  }
}
