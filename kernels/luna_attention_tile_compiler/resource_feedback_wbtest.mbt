///|
fn feedback_test_budget() -> AttentionResourceBudget raise {
  AttentionResourceBudget::new(
    units=36,
    threads_per_unit=1536,
    registers_per_unit=65536,
    register_allocation_grain=256,
    register_allocation_threads=32,
    shared_bytes_per_unit=102400L,
    blocks_per_unit=24,
  )
}

///|
test "rounded waves expose the last incomplete wave" {
  assert_eq(attention_wave_score(3600L, 36L, 36L), 100L)
  assert_eq(attention_wave_score(3700L, 37L, 36L), 200L)
  assert_eq(attention_wave_score(7200L, 72L, 36L), 200L)
  assert_eq(attention_wave_score(7300L, 73L, 36L), 300L)
  assert_eq(
    attention_wave_score(9223372036854775807L, 1L, 36L),
    9223372036854775807L,
  )
  for jobs in 1L..<200L {
    for slots in 1L..<100L {
      let base = jobs * 101L + 17L
      let waves = 1L + (jobs - 1L) / slots
      assert_eq(
        attention_wave_score(base, jobs, slots),
        (base * waves + jobs - 1L) / jobs,
      )
    }
  }
}

///|
test "resource feedback is immutable and unknown registers are not zero" {
  let base = functional_compiler_request([])
  let request = base.with_resource_feedback(feedback_test_budget(), [])
  let frontier = compile_attention_tile_frontier(request)
  assert_true(
    frontier.variants().length() >
    compile_attention_tile_frontier(base).variants().length(),
  )
  let variant = frontier.variants()[0]
  let absent = attention_tile_resource_score(request, variant)
  let measured = request.with_resource_feedback(feedback_test_budget(), [
    AttentionRegisterObservation::new(
      variant.schedule().digest(),
      registers_per_thread=16,
    ),
  ])
  assert_true(attention_tile_resource_score(measured, variant) <= absent)
  assert_eq(compile_attention_tile_frontier(request), frontier)
  assert_true(base.resource_budget is None)
}

///|
test "offline observations canonicalize in schedule order" {
  let base = functional_compiler_request([]).with_resource_feedback(
    feedback_test_budget(),
    [],
  )
  let variants = compile_attention_tile_frontier(base).variants()
  let a = AttentionRegisterObservation::new(
    variants[0].schedule().digest(),
    registers_per_thread=32,
  )
  let b = AttentionRegisterObservation::new(
    variants[1].schedule().digest(),
    registers_per_thread=168,
  )
  let first = compile_attention_tile_frontier(
    base.with_resource_feedback(feedback_test_budget(), [a, b]),
  )
  let reversed = compile_attention_tile_frontier(
    base.with_resource_feedback(feedback_test_budget(), [b, a]),
  )
  assert_eq(first, reversed)
  try base.with_resource_feedback(feedback_test_budget(), [a, a]) catch {
    InvalidAttentionTileCompilation(ResourceFeedback) => ()
    _ => fail("wrong duplicate error")
  } noraise {
    _ => fail("duplicate observation accepted")
  }
}

///|
test "numerical and read-view contracts participate in program identity" {
  let base = functional_compiler_request([])
  let problem = base.problem()
  let (q, k) = problem.dimensions()
  let changed = @luna_attention_strategy.AttentionCompileProblem::new(
    shape=problem.shape(),
    phase=problem.phase(),
    numeric_type=problem.numeric_type(),
    batch_rows=problem.batch_rows(),
    query_rows=q,
    context_tokens=k,
    causal=problem.is_causal(),
    allow_alternative_softmax_reduction=true,
    key_value_read_view=DenseCurrentAndPagedHistory,
  )
  let candidates = @luna_attention_strategy.generate_attention_tile_candidates(
    problem,
    base.capabilities,
  )
  let first = compile_static_attention_tile_candidate(base, candidates[0])
  let second = compile_static_attention_tile_candidate(
    { ..base, problem: changed, },
    candidates[0],
  )
  assert_true(first.program().digest() != second.program().digest())
}

///|
test "partition map selection also consumes immutable resource feedback" {
  let request = functional_compiler_request([], query_rows=64).with_resource_feedback(
    feedback_test_budget(),
    [],
  )
  let compiled = compile_partitioned_attention_tile(request)
  assert_true(compiled.schedule().key_value_partitions() > 1)
  assert_eq(compiled, compile_partitioned_attention_tile(request))
  assert_true(
    compiled.optimized.semantic.selected.request.resource_budget is Some(_),
  )
}

///|
test "no resident workgroup is not a selectable schedule" {
  let impossible = AttentionResourceBudget::new(
    units=1,
    threads_per_unit=1,
    registers_per_unit=1,
    register_allocation_grain=1,
    shared_bytes_per_unit=1L,
    blocks_per_unit=1,
  )
  let request = functional_compiler_request([]).with_resource_feedback(
    impossible,
    [],
  )
  try compile_attention_tile(request) catch {
    InvalidAttentionTileCompilation(ResourceFeedback) => ()
    _ => fail("unexpected resource error")
  } noraise {
    _ => fail("impossible residency accepted")
  }
}
