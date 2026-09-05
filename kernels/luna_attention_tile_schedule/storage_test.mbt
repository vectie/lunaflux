///|
test "public storage layout preserves portable ranges through scheduling" {
  let problem = @luna_attention_strategy.AttentionCompileProblem::new(
    shape=@luna_attention_strategy.AttentionShape::new(
      query_heads=16,
      kv_heads=4,
      head_dimension=64,
      tokens_per_page=16,
    ),
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
    max_pipeline_stages=1,
    supports_async_copy=false,
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
    target=@luna_attention_strategy.AttentionTargetId::new("storage-test"),
    problem~,
    capabilities~,
  )
  let program = @luna_attention_tile_ir.compile_attention_tile_ir(plan)
  let optimization = @luna_attention_tile_optimizer.optimize_attention_tile_program(
    program,
  )
  let scheduled = @luna_attention_tile_schedule.schedule_rewritten_attention_tile_program(
    program, optimization,
  )
  guard scheduled.matrix_storage_layout() is Some(layout) else {
    fail("expected matrix storage plan")
  }
  for span in layout.spans() {
    assert_eq(layout.offset(span.role()), span.range().0)
    assert_true(span.range().0 + span.range().1 <= layout.peak_bytes())
    assert_true(span.lifetime().0 < span.lifetime().1)
  }
  assert_eq(layout.peak_bytes(), scheduled.execution_geometry().3)
}
