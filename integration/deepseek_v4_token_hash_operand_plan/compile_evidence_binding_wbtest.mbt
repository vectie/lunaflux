///|
test "compile evidence cannot promote unavailable Query-B projection" {
  let requirements = operand_plan_test_requirements()
  let candidate = operand_plan_test_candidate(requirements)
  assert_artifact_binding_test_query_b_operation_unavailable(
    requirements, candidate,
  )
}
