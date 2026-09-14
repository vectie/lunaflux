///|
test "long batch growth requires matching physical and flattened page capacity" {
  let scheduler = scheduler_fixture(
    step_token_budget=2048,
    max_active_requests=32,
    max_waiting_requests=32,
    prefill_chunk_tokens=2048,
    output_event_capacity=16384,
  )
  let cache = cache_fixture(
    tokens_per_page=8,
    total_page_count=9216,
    block_table_pages_per_request=5120,
    max_active_references_per_page=1,
  )
  let shapes = shape_fixture(
    max_batch_rows=32,
    max_sequence_tokens=40960,
    vocabulary_size=151936,
  )
  for entries in [8192, 9216] {
    let worker = worker_fixture(
      max_prefill_rows=32,
      max_decode_rows=32,
      max_plan_rows=32,
      max_plan_tokens=2048,
      max_plan_pages=entries,
      max_completion_slots=32,
      max_sequence_tokens=40960,
      max_token_id=151935,
    )
    if entries == 8192 {
      expect_capacity_error(
        () => @runtime_resolved.resolve(scheduler, cache, shapes, worker),
        Invalid(WorkerPlanPages),
      )
    } else {
      let capacity = @runtime_resolved.resolve(scheduler, cache, shapes, worker)
      assert_eq(capacity.required_worker_page_table_cells(), 9216)
      assert_eq(capacity.total_page_count(), 9216)
    }
  }
}
