///|
fn schedule_test_program() -> @luna_attention_tile_ir.AttentionTileProgram raise {
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
    query_rows=128,
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
  assert_eq(first.loops()[0].geometry(), (128, 16))
  assert_eq(first.loops()[1].geometry(), (16, 1))
  assert_eq(first.loops()[2].kind(), AttentionSequentialFold)
  assert_eq(first.loops()[2].axis(), AttentionKeyValueAxis)
  assert_eq(first.loops()[2].geometry(), (4096, 128))
  assert_eq(first.loops()[3].kind(), AttentionVectorMap)
  assert_eq(first.loops()[3].geometry(), (128, 8))
  assert_eq(first.compute(), AttentionMatrixDot)
  assert_eq(first.memory(), AttentionAsyncStagedShared)
  assert_eq(first.execution_geometry(), (8, 3, 16, 208896L))
  assert_eq(first.digest().as_hex().length(), 64)
}
