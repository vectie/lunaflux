///|
fn compiler_test_shape() -> AttentionShape raise AttentionStrategyError {
  AttentionShape::new(
    query_heads=16,
    kv_heads=4,
    head_dimension=128,
    tokens_per_page=16,
  )
}

///|
fn compiler_test_problem() -> AttentionCompileProblem raise AttentionStrategyError {
  AttentionCompileProblem::new(
    shape=compiler_test_shape(),
    phase=Prefill,
    numeric_type=AttentionBFloat16,
    query_rows=128,
    context_tokens=4096,
    causal=true,
  )
}

///|
fn compiler_test_capabilities() -> AttentionCompilerCapabilities raise AttentionStrategyError {
  AttentionCompilerCapabilities::new(
    subgroup_width=32,
    max_threads_per_block=256,
    max_shared_memory_bytes=262144L,
    max_pipeline_stages=3,
    supports_async_copy=true,
    matrix_shape=Some(
      AttentionMatrixShape::new(rows=16, columns=16, reduction=16),
    ),
  )
}

///|
fn compiler_candidate_with_id(
  candidates : ReadOnlyArray[AttentionTileCandidate],
  stable_id : Int,
) -> AttentionTileCandidate raise {
  for candidate in candidates {
    if candidate.stable_id() == stable_id {
      return candidate
    }
  }
  fail("missing compiler candidate")
}

///|
fn compiler_record(
  backend : AttentionBackendId,
  target : AttentionTargetId,
  problem : AttentionCompileProblem,
  candidate : AttentionTileCandidate,
  latency : Int64,
) -> AttentionAutotuneRecord raise AttentionStrategyError {
  AttentionAutotuneRecord::new(
    key=AttentionAutotuneKey::new(backend~, target~, problem~, candidate~),
    median_latency_ns=latency,
    sample_count=9,
  )
}

///|
fn expect_attention_compiler_rule(
  body : () -> Unit raise,
  expected : AttentionStrategyRule,
) -> Unit raise {
  try body() catch {
    AttentionStrategyError::InvalidAttentionStrategy(rule) if rule == expected =>
      ()
    _ => fail("unexpected attention compiler rejection")
  } noraise {
    _ => fail("expected attention compiler rejection")
  }
}

///|
test "attention compiler generates portable long-prefill tile candidates" {
  let problem = compiler_test_problem()
  let candidates = generate_attention_tile_candidates(
    problem,
    compiler_test_capabilities(),
  )
  assert_eq(candidates.length(), 12)
  let matrix_narrow = compiler_candidate_with_id(candidates, 290)
  assert_eq(matrix_narrow.compute(), AttentionMatrixDot)
  assert_eq(matrix_narrow.memory(), AttentionStagedShared)
  assert_eq(matrix_narrow.tile().dimensions(), (16, 32, 128))
  assert_eq(matrix_narrow.execution_geometry(), (4, 1, 16))
  assert_eq(matrix_narrow.shared_memory_bytes(), 32976L)
  let dual_matrix = compiler_candidate_with_id(candidates, 310)
  assert_eq(dual_matrix.compute(), AttentionMatrixQkPv)
  assert_eq(dual_matrix.tile().dimensions(), (16, 32, 128))
  assert_eq(dual_matrix.execution_geometry(), (8, 1, 16))
  assert_eq(dual_matrix.shared_memory_bytes(), 31952L)
  let matrix_async = compiler_candidate_with_id(candidates, 301)
  assert_eq(matrix_async.compute(), AttentionMatrixDot)
  assert_eq(matrix_async.memory(), AttentionAsyncStagedShared)
  assert_eq(matrix_async.tile().dimensions(), (16, 128, 128))
  assert_eq(matrix_async.execution_geometry(), (8, 3, 16))
  assert_eq(matrix_async.shared_memory_bytes(), 225488L)
  let plan = compile_attention_tile_plan(
    [],
    backend=AttentionBackendId::new("portable"),
    target=AttentionTargetId::new("fixture"),
    problem~,
    capabilities=compiler_test_capabilities(),
  )
  assert_eq(plan.source(), AttentionStaticCostModel)
  assert_eq(plan.candidate().stable_id(), 301)
}

///|
test "attention compiler prunes unsupported pipelines and matrix operations" {
  let problem = compiler_test_problem()
  let constrained = AttentionCompilerCapabilities::new(
    subgroup_width=32,
    max_threads_per_block=128,
    max_shared_memory_bytes=65536L,
    max_pipeline_stages=2,
    supports_async_copy=false,
  )
  let candidates = generate_attention_tile_candidates(problem, constrained)
  assert_eq(candidates.length(), 3)
  assert_eq(candidates[0].stable_id(), 100)
  assert_eq(candidates[1].stable_id(), 200)
  assert_eq(candidates[2].stable_id(), 210)
  let plan = compile_attention_tile_plan(
    [],
    backend=AttentionBackendId::new("portable"),
    target=AttentionTargetId::new("small-shared-memory"),
    problem~,
    capabilities=constrained,
  )
  assert_eq(plan.candidate().stable_id(), 210)
}

///|
test "attention autotune overrides the cost model without leaking device names" {
  let backend = AttentionBackendId::new("portable")
  let target = AttentionTargetId::new("device-a")
  let problem = compiler_test_problem()
  let capabilities = compiler_test_capabilities()
  let candidates = generate_attention_tile_candidates(problem, capabilities)
  let subgroup = compiler_candidate_with_id(candidates, 220)
  let matrix = compiler_candidate_with_id(candidates, 300)
  let slower = compiler_record(backend, target, problem, matrix, 120L)
  let faster = compiler_record(backend, target, problem, subgroup, 100L)
  let first = compile_attention_tile_plan(
    [slower, faster],
    backend~,
    target~,
    problem~,
    capabilities~,
  )
  let second = compile_attention_tile_plan(
    [faster, slower],
    backend~,
    target~,
    problem~,
    capabilities~,
  )
  assert_eq(first.source(), AttentionOfflineAutotune)
  assert_eq(first.candidate().stable_id(), 220)
  assert_eq(second.candidate().stable_id(), 220)
  let foreign = compile_attention_tile_plan(
    [faster],
    backend~,
    target=AttentionTargetId::new("device-b"),
    problem~,
    capabilities~,
  )
  assert_eq(foreign.source(), AttentionStaticCostModel)
  assert_eq(foreign.candidate().stable_id(), 301)
}

///|
test "attention autotune rejects duplicate and aliased candidate identity" {
  let backend = AttentionBackendId::new("portable")
  let target = AttentionTargetId::new("device-a")
  let problem = compiler_test_problem()
  let capabilities = compiler_test_capabilities()
  let candidate = compiler_candidate_with_id(
    generate_attention_tile_candidates(problem, capabilities),
    220,
  )
  let record = compiler_record(backend, target, problem, candidate, 100L)
  expect_attention_compiler_rule(
    () => {
      ignore(
        compile_attention_tile_plan(
          [record, record],
          backend~,
          target~,
          problem~,
          capabilities~,
        ),
      )
    },
    Duplicate,
  )
  let aliased = {
    ..candidate,
    memory: AttentionDirectGlobal,
    shared_memory_bytes: 0L,
  }
  let conflict = compiler_record(backend, target, problem, aliased, 90L)
  expect_attention_compiler_rule(
    () => {
      ignore(
        compile_attention_tile_plan(
          [record, conflict],
          backend~,
          target~,
          problem~,
          capabilities~,
        ),
      )
    },
    Conflict,
  )
}

///|
test "attention compile problem separates prefill and decode invariants" {
  expect_attention_compiler_rule(
    () => {
      ignore(
        AttentionCompileProblem::new(
          shape=compiler_test_shape(),
          phase=Decode,
          numeric_type=AttentionBFloat16,
          query_rows=2,
          context_tokens=128,
          causal=true,
        ),
      )
    },
    Problem,
  )
}
