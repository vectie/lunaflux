///|
fn scheduler_fixture(
  step_token_budget? : Int = 128,
  max_active_requests? : Int = 8,
  max_waiting_requests? : Int = 32,
  prefill_chunk_tokens? : Int = 64,
  emergency_decode_page_reserve? : Int = 8,
) -> @scheduler.SchedulerConfig {
  try! @scheduler.new(
    step_token_budget~,
    max_active_requests~,
    max_waiting_requests~,
    prefill_chunk_tokens~,
    emergency_decode_page_reserve~,
    waiting_age_threshold_steps=128,
  )
}

///|
fn cache_fixture(
  tokens_per_page? : Int = 16,
  total_page_count? : Int = 4096,
  block_table_pages_per_request? : Int = 256,
) -> @cache.CacheConfig {
  try! @cache.new(
    tokens_per_page~,
    total_page_count~,
    block_table_pages_per_request~,
    prefix_enabled=false,
    max_prefix_entries=0,
    max_prefix_nodes=0,
    max_prefix_tokens_per_entry=0,
    max_prefix_pages=0,
    max_active_references_per_page=1024,
    max_cached_references_per_page=1024,
    layout_version=1,
  )
}

///|
fn shape_fixture(
  max_batch_rows? : Int = 8,
  max_sequence_tokens? : Int = 4096,
  vocabulary_size? : Int = 32000,
) -> @plan.ShapeConstraints {
  try! @plan.ShapeConstraints::new(
    max_batch_rows~,
    max_sequence_tokens~,
    vocabulary_size~,
    hidden_size=64,
    intermediate_size=256,
    attention_head_count=8,
    kv_head_count=2,
    head_dimension=8,
  )
}

///|
fn worker_fixture(
  max_prefill_rows? : Int = 8,
  max_decode_rows? : Int = 8,
  max_plan_rows? : Int = 8,
  max_plan_tokens? : Int = 128,
  max_plan_pages? : Int = 2048,
  max_completion_slots? : Int = 8,
  max_sequence_tokens? : Int = 4096,
  max_token_id? : Int = 31999,
) -> @worker_protocol.WorkerProtocolLimits {
  try! @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows~,
    max_decode_rows~,
    max_plan_rows~,
    max_plan_tokens~,
    max_plan_pages~,
    max_capabilities=256,
    max_completion_slots~,
    max_sequence_tokens~,
    max_token_id~,
  )
}

///|
fn expect_capacity_error(
  run : () -> @runtime_resolved.ResolvedRuntimeCapacity raise @runtime_resolved.RuntimeCapacityError,
  expected : @runtime_resolved.RuntimeCapacityError,
) -> Unit raise {
  try run() catch {
    error => assert_eq(error, expected)
  } noraise {
    _ => fail("expected runtime capacity rejection")
  }
}
