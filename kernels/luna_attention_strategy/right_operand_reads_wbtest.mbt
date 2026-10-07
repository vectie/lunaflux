///|
test "read schedules preserve static choice and separate exact measured identities" {
  let shape = AttentionShape::new(
    query_heads=16,
    kv_heads=4,
    head_dimension=128,
    tokens_per_page=16,
  )
  let problem = AttentionCompileProblem::new(
    shape~,
    phase=Prefill,
    numeric_type=AttentionBFloat16,
    query_rows=2048,
    context_tokens=32768,
    causal=true,
    allow_alternative_softmax_reduction=true,
    allow_approximate_exponential=true,
    required_fragment_ownership=Some(QueryOwnedFragments),
  )
  let capabilities = AttentionCompilerCapabilities::new(
    subgroup_width=32,
    max_threads_per_block=256,
    max_shared_memory_bytes=100000L,
    max_pipeline_stages=2,
    supports_async_copy=true,
    matrix_shape=Some(
      AttentionMatrixShape::new(rows=16, columns=16, reduction=16),
    ),
  )
  let candidates = generate_attention_tile_candidates(problem, capabilities)
  let base = candidates.filter(c => c.stable_id() == 30322)[0]
  let backend = AttentionBackendId::new("cuda")
  let target = AttentionTargetId::new("measured-device")
  assert_eq(
    compile_attention_tile_plan([], backend~, target~, problem~, capabilities~)
    .candidate()
    .right_operand_read_slots(),
    0,
  )
  for slots in [1, 2] {
    let candidate = candidates.filter(c => {
        c.stable_id() == (slots + 1) * 100000 + 30322
      })[0]
    assert_eq(candidate.right_operand_read_slots(), slots)
    assert_eq(candidate.tile(), base.tile())
    assert_eq(
      attention_candidate_static_cost(problem, candidate),
      attention_candidate_static_cost(problem, base),
    )
    assert_true(
      attention_candidate_requires_approximate_exponential(
        candidate.stable_id(),
      ),
    )
    let record = AttentionAutotuneRecord::new(
      key=AttentionAutotuneKey::new(backend~, target~, problem~, candidate~),
      median_latency_ns=100L,
      sample_count=5,
    )
    assert_eq(
      compile_attention_tile_plan(
        [record],
        backend~,
        target~,
        problem~,
        capabilities~,
      ).candidate(),
      candidate,
    )
    assert_eq(
      compile_attention_tile_plan(
        [record],
        backend~,
        target=AttentionTargetId::new("other-device"),
        problem~,
        capabilities~,
      )
      .candidate()
      .right_operand_read_slots(),
      0,
    )
  }
}
