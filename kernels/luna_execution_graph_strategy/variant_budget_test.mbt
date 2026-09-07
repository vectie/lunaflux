///|
test "optional graph reservation is immutable and bounded on both axes" {
  let budget = ExecutionGraphVariantBudget::new(2, 7)
  assert_eq(budget.reserve(0), None)
  assert_eq(budget.reserve(-1), None)
  assert_eq(budget.reserve(8), None)
  let next = budget.reserve(3).unwrap()
  assert_eq(next.reserve(5), None)
  let exhausted = next.reserve(4).unwrap()
  assert_eq(exhausted.reserve(1), None)
  assert_eq(budget.reserve(3), Some(next))
  let graph_limited = ExecutionGraphVariantBudget::new(1, 100)
  assert_eq(graph_limited.reserve(1).unwrap().reserve(1), None)
  assert_eq(ExecutionGraphVariantBudget::new(0, 100).reserve(1), None)
}
