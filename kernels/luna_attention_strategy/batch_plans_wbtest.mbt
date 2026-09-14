///|
test "single enumeration plans equal individually validated plans" {
  let problem = compiler_test_problem()
  let capabilities = compiler_test_capabilities()
  let plans = compile_attention_tile_candidate_plans(problem, capabilities)
  let candidates = generate_attention_tile_candidates(problem, capabilities)
  assert_eq(plans.length(), candidates.length())
  for index, plan in plans {
    assert_eq(
      plan,
      compile_attention_tile_candidate_plan(
        problem,
        capabilities,
        candidates[index],
      ),
    )
  }
}
