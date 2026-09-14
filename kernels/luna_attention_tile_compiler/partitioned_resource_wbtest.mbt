///|
test "measured partitioned winner must fit the current resource budget" {
  let base = functional_compiler_request([], query_rows=64)
  let selected = select_attention_tile(base)
  let record = @luna_attention_strategy.AttentionAutotuneRecord::new(
    key=@luna_attention_strategy.AttentionAutotuneKey::new(
      backend=base.backend,
      target=base.target,
      problem=base.problem,
      candidate=selected.plan().candidate(),
    ),
    median_latency_ns=100L,
    sample_count=7,
  )
  let measured = functional_compiler_request([record], query_rows=64)
  let budget = AttentionResourceBudget::new(
    units=36,
    threads_per_unit=1024,
    registers_per_unit=65536,
    register_allocation_grain=256,
    register_allocation_threads=32,
    shared_bytes_per_unit=100000L,
    blocks_per_unit=16,
  )
  let reference = compile_partitioned_attention_tile(measured)
  let observation = AttentionRegisterObservation::new(
    reference.schedule().digest(),
    registers_per_thread=32,
  )
  for
    impossible_budget in [
      { ..budget, shared_bytes_per_unit: 1L, },
      { ..budget, threads_per_unit: 32, },
      { ..budget, registers_per_unit: 256, },
    ] {
    let impossible = measured.with_resource_feedback(impossible_budget, [
      observation,
    ])
    try compile_partitioned_attention_tile(impossible) catch {
      InvalidAttentionTileCompilation(ResourceFeedback) => ()
      _ => fail("unexpected partitioned resource failure")
    } noraise {
      _ => fail("measured split-K winner bypassed the resource budget")
    }
  }
  let supported = measured.with_resource_feedback(budget, [observation])
  let compiled = compile_partitioned_attention_tile(supported)
  assert_eq(compiled.plan().source(), AttentionOfflineAutotune)
  assert_eq(compiled.plan().candidate(), selected.plan().candidate())
  assert_eq(compiled.schedule().key_value_partitions(), 2)
  assert_eq(compiled.program(), reference.program())
  assert_eq(compiled.schedule(), reference.schedule())
  assert_eq(compiled.digest(), reference.digest())
}
