///|
fn schedule_test_program(
  query_rows? : Int = 128,
) -> @luna_attention_tile_ir.AttentionTileProgram raise {
  let shape = @luna_attention_strategy.AttentionShape::new(
    query_heads=16,
    kv_heads=4,
    head_dimension=128,
    tokens_per_page=16,
  )
  let problem = @luna_attention_strategy.AttentionCompileProblem::new(
    shape~,
    phase=Prefill,
    numeric_type=AttentionBFloat16,
    query_rows~,
    context_tokens=4096,
    causal=true,
  )
  let capabilities = @luna_attention_strategy.AttentionCompilerCapabilities::new(
    subgroup_width=32,
    max_threads_per_block=256,
    max_shared_memory_bytes=262144L,
    max_pipeline_stages=3,
    supports_async_copy=true,
    matrix_shape=Some(
      @luna_attention_strategy.AttentionMatrixShape::new(
        rows=16,
        columns=16,
        reduction=16,
      ),
    ),
  )
  let plan = @luna_attention_strategy.compile_attention_tile_plan(
    [],
    backend=@luna_attention_strategy.AttentionBackendId::new("portable"),
    target=@luna_attention_strategy.AttentionTargetId::new("fixture"),
    problem~,
    capabilities~,
  )
  @luna_attention_tile_ir.compile_attention_tile_ir(plan)
}

///|
test "attention schedule exposes portable parallel map and online fold" {
  let program = schedule_test_program()
  let first = schedule_attention_tile_program(program)
  let repeated = schedule_attention_tile_program(program)
  assert_eq(first.digest(), repeated.digest())
  assert_eq(first.canonical_bytes(), repeated.canonical_bytes())
  assert_eq(first.program_digest(), program.digest())
  assert_eq(first.loops().length(), 4)
  assert_eq(first.loops()[0].kind(), AttentionParallelMap)
  assert_eq(first.loops()[0].axis(), AttentionQueryRowAxis)
  assert_eq(first.loops()[0].geometry(), (128, 64))
  assert_eq(first.loops()[1].geometry(), (16, 1))
  assert_eq(first.loops()[2].kind(), AttentionSequentialFold)
  assert_eq(first.loops()[2].axis(), AttentionKeyValueAxis)
  assert_eq(first.loops()[2].geometry(), (4096, 64))
  assert_eq(first.loops()[3].kind(), AttentionVectorMap)
  assert_eq(first.loops()[3].geometry(), (128, 8))
  assert_eq(first.compute(), AttentionMatrixQkPv)
  assert_eq(first.memory(), AttentionStagedShared)
  assert_eq(first.execution_geometry(), (8, 1, 16, 107280L))
  assert_false(first.hoists_paged_key_value_page_lookup())
  assert_false(first.reuses_online_softmax_tile_storage())
  assert_eq(first.digest().as_hex().length(), 64)
}

///|
test "long prefill schedule exposes parallel KV partitions and associative merge" {
  let program = schedule_test_program(query_rows=64)
  let optimization = @luna_attention_tile_optimizer.optimize_attention_tile_program(
    program,
  )
  let schedule = schedule_optimized_attention_tile_program(
    program, optimization,
  )
  assert_eq(schedule.key_value_partitions(), 2)
  assert_true(schedule.hoists_paged_key_value_page_lookup())
  assert_true(schedule.reuses_online_softmax_tile_storage())
  assert_eq(schedule.execution_geometry().3, 98304L)
  assert_eq(schedule.loops().length(), 6)
  assert_eq(schedule.loops()[2].kind(), AttentionParallelMap)
  assert_eq(schedule.loops()[2].axis(), AttentionKeyValuePartitionAxis)
  assert_eq(schedule.loops()[2].geometry(), (4096, 2048))
  assert_eq(schedule.loops()[3].kind(), AttentionSequentialFold)
  assert_eq(schedule.loops()[3].geometry(), (2048, 64))
  assert_eq(schedule.loops()[4].kind(), AttentionAssociativeFold)
  assert_eq(schedule.loops()[4].axis(), AttentionSoftmaxPartitionAxis)
  assert_eq(schedule.loops()[4].geometry(), (2, 1))
  assert_eq(schedule.loops()[5].kind(), AttentionVectorMap)
}

///|
test "single-fold rewritten schedule preserves optimizer page lookup hoist" {
  let program = schedule_test_program()
  let optimization = @luna_attention_tile_optimizer.optimize_attention_tile_program(
    program,
  )
  let schedule = schedule_rewritten_attention_tile_program(
    program, optimization,
  )
  assert_eq(schedule.key_value_partitions(), 1)
  assert_true(schedule.hoists_paged_key_value_page_lookup())
  assert_true(schedule.reuses_online_softmax_tile_storage())
  assert_true(
    @utf8.decode(schedule.canonical_bytes()).contains(
      "hoist_paged_key_value_page_lookup=true",
    ),
  )
}
