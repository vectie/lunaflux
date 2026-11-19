///|
test "embedding scatter has causal indexing and disjoint vector ownership" {
  let plan = @elementwise_physical_ir.EmbeddingScatterProgram::new(
    width=8,
    maximum_rows=3,
    maximum_positions=16,
    maximum_features=5,
    owners=4,
  ).unwrap()
  assert_eq(plan.geometry(), (8, 3, 16, 5))
  assert_eq(plan.ownership().geometry(), (8, 4, 2))
  assert_eq(plan.ownership().vector_at(2, 1), Some(6))
  assert_true(
    @elementwise_physical_ir.EmbeddingScatterProgram::new(
      width=8,
      maximum_rows=17,
      maximum_positions=16,
      maximum_features=5,
      owners=4,
    )
    is None,
  )
  assert_true(
    @elementwise_physical_ir.EmbeddingScatterProgram::new(
      width=8,
      maximum_rows=3,
      maximum_positions=16,
      maximum_features=17,
      owners=4,
    )
    is None,
  )
}
