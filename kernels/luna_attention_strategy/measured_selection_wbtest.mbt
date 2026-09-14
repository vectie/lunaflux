///|
test "measured selection retains exact candidate cost and deterministic ties" {
  let backend = AttentionBackendId::new("portable")
  let target = AttentionTargetId::new("device-a")
  let problem = compiler_test_problem()
  let capabilities = compiler_test_capabilities()
  let candidates = generate_attention_tile_candidates(problem, capabilities)
  for candidate in candidates {
    let record = compiler_record(backend, target, problem, candidate, 100L)
    let plan = compile_attention_tile_plan(
      [record],
      backend~,
      target~,
      problem~,
      capabilities~,
    )
    assert_eq(plan.candidate(), candidate)
    assert_eq(plan.source(), AttentionOfflineAutotune)
    assert_eq(
      plan.static_cost(),
      attention_candidate_static_cost(problem, candidate),
    )
  }
  let a = candidates[0]
  let b = candidates[candidates.length() - 1]
  let ra = compiler_record(backend, target, problem, a, 100L)
  let rb = compiler_record(backend, target, problem, b, 100L)
  let first = compile_attention_tile_plan(
    [ra, rb],
    backend~,
    target~,
    problem~,
    capabilities~,
  )
  let second = compile_attention_tile_plan(
    [rb, ra],
    backend~,
    target~,
    problem~,
    capabilities~,
  )
  assert_eq(first, second)
  assert_eq(
    first.candidate().stable_id(),
    Int::min(a.stable_id(), b.stable_id()),
  )
}
