///|
fn scheduler_fixture(
  step_token_budget~ : Int32 = 128,
  max_active_requests~ : Int32 = 8,
  max_waiting_requests~ : Int32 = 32,
  prefill_chunk_tokens~ : Int32 = 64,
  emergency_decode_page_reserve~ : Int32 = 8,
  output_event_capacity~ : Int32 = 128,
) -> @scheduler.SchedulerConfig {
  try! @scheduler.new(
    step_token_budget~,
    max_active_requests~,
    max_waiting_requests~,
    prefill_chunk_tokens~,
    emergency_decode_page_reserve~,
    waiting_age_threshold_steps=128,
    output_event_capacity~,
  )
}

///|
fn cache_fixture(
  tokens_per_page~ : Int32 = 16,
  total_page_count~ : Int32 = 4096,
  block_table_pages_per_request~ : Int32 = 256,
  max_active_references_per_page~ : Int32 = 1024,
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
    max_active_references_per_page~,
    max_cached_references_per_page=1024,
    layout_version=1,
  )
}

///|
fn prefix_cache_fixture(
  max_prefix_entries~ : Int32 = 64,
  max_prefix_nodes~ : Int32 = 1024,
  max_prefix_tokens_per_entry~ : Int32 = 256,
  max_prefix_pages~ : Int32 = 256,
  max_prefix_scope_bytes~ : Int32 = 64,
  max_active_references_per_page~ : Int32 = 1024,
) -> @cache.CacheConfig {
  try! @cache.new(
    tokens_per_page=16,
    total_page_count=4096,
    block_table_pages_per_request=256,
    prefix_enabled=true,
    max_prefix_entries~,
    max_prefix_nodes~,
    max_prefix_tokens_per_entry~,
    max_prefix_pages~,
    max_prefix_scope_bytes~,
    max_active_references_per_page~,
    max_cached_references_per_page=1024,
    layout_version=7,
  )
}

///|
fn shape_fixture(
  max_batch_rows~ : Int32 = 8,
  max_sequence_tokens~ : Int32 = 4096,
  vocabulary_size~ : Int32 = 32000,
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
  max_prefill_rows~ : Int32 = 8,
  max_decode_rows~ : Int32 = 8,
  max_plan_rows~ : Int32 = 8,
  max_plan_tokens~ : Int32 = 128,
  max_plan_pages~ : Int32 = 2048,
  max_completion_slots~ : Int32 = 8,
  max_sequence_tokens~ : Int32 = 4096,
  max_token_id~ : Int32 = 31999,
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
