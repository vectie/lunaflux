///|
test "realistic startup capacity resolves exact component envelopes" {
  let capacity = @runtime_resolved.resolve(
    scheduler_fixture(),
    cache_fixture(),
    shape_fixture(),
    worker_fixture(),
  )
  assert_eq(capacity.tokens_per_page(), 16)
  assert_eq(capacity.total_page_count(), 4096)
  assert_eq(capacity.context_pages_per_request(), 256)
  assert_eq(capacity.block_table_pages_per_request(), 256)
  assert_eq(capacity.max_active_requests(), 8)
  assert_eq(capacity.max_waiting_requests(), 32)
  assert_eq(capacity.total_request_slots(), 40)
  assert_eq(capacity.step_token_budget(), 128)
  assert_eq(capacity.prefill_chunk_tokens(), 64)
  assert_eq(capacity.emergency_decode_page_reserve(), 8)
  assert_eq(capacity.waiting_age_threshold_steps(), 128)
  assert_eq(capacity.max_rows_per_step(), 8)
  assert_eq(capacity.required_worker_page_table_cells(), 2048)
  assert_eq(capacity.plan_buffer_count(), 2)
  assert_eq(capacity.worker_max_capabilities(), 256)
  let page_limits = capacity.page_allocator_limits()
  assert_eq(page_limits.page_capacity(), 4096)
  assert_eq(page_limits.max_run_pages(), 256)
  assert_eq(page_limits.max_active_references_per_page(), 1024)
  assert_eq(page_limits.max_cached_references_per_page(), 1024)
  assert_eq(page_limits.max_generation(), 2_147_483_647)
  let table_limits = capacity.block_table_limits()
  assert_eq(table_limits.table_capacity(), 8)
  assert_eq(table_limits.max_pages_per_table(), 256)
  assert_eq(table_limits.tokens_per_page(), 16)
  assert_eq(table_limits.max_generation(), 2_147_483_647)
}

///|
test "active concurrency may exceed one submitted model batch" {
  let capacity = @runtime_resolved.resolve(
    scheduler_fixture(
      step_token_budget=8,
      max_active_requests=64,
      prefill_chunk_tokens=8,
    ),
    cache_fixture(),
    shape_fixture(max_batch_rows=8),
    worker_fixture(),
  )
  assert_eq(capacity.max_active_requests(), 64)
  assert_eq(capacity.max_rows_per_step(), 8)
  assert_eq(capacity.block_table_limits().table_capacity(), 64)
}

///|
test "ceiling context pages accepts the exact non-divisible boundary" {
  let capacity = @runtime_resolved.resolve(
    scheduler_fixture(),
    cache_fixture(block_table_pages_per_request=257),
    shape_fixture(max_sequence_tokens=4097),
    worker_fixture(max_plan_pages=2056, max_sequence_tokens=4097),
  )
  assert_eq(capacity.context_pages_per_request(), 257)
  assert_eq(capacity.required_worker_page_table_cells(), 2056)
}

///|
test "identical startup inputs resolve equal values and decisions" {
  let first = @runtime_resolved.resolve(
    scheduler_fixture(),
    cache_fixture(),
    shape_fixture(),
    worker_fixture(),
  )
  let second = @runtime_resolved.resolve(
    scheduler_fixture(),
    cache_fixture(),
    shape_fixture(),
    worker_fixture(),
  )
  assert_eq(first, second)
  let decisions = first.decisions()
  assert_eq(decisions.length(), 7)
  assert_eq(decisions[0].subject(), ContextPagesPerRequest)
  assert_eq(decisions[0].reason(), CeilingDivision)
  assert_eq(decisions[2].subject(), MaxRowsPerStep)
  assert_eq(decisions[2].reason(), MinimumOfActiveAndTokenBudget)
  assert_eq(decisions[4].reason(), FixedDoubleBuffer)
  assert_eq(decisions[5].reason(), OwnerMaximumGenerationDefault)
}

///|
test "physical page reserve cannot exceed the arena" {
  expect_capacity_error(
    () => {
      @runtime_resolved.resolve(
        scheduler_fixture(emergency_decode_page_reserve=8),
        cache_fixture(total_page_count=4, block_table_pages_per_request=4),
        shape_fixture(max_sequence_tokens=64),
        worker_fixture(max_plan_pages=32, max_sequence_tokens=64),
      )
    },
    Invalid(EmergencyReserveWithinPhysicalPages),
  )
}

///|
test "model context must fit each request block table" {
  expect_capacity_error(
    () => {
      @runtime_resolved.resolve(
        scheduler_fixture(),
        cache_fixture(block_table_pages_per_request=256),
        shape_fixture(max_sequence_tokens=4097),
        worker_fixture(max_sequence_tokens=4097),
      )
    },
    Invalid(ContextFitsBlockTable),
  )
}

///|
test "worker token and sequence envelopes cover scheduler and model" {
  expect_capacity_error(
    () => {
      @runtime_resolved.resolve(
        scheduler_fixture(),
        cache_fixture(),
        shape_fixture(),
        worker_fixture(max_plan_tokens=127),
      )
    },
    Invalid(WorkerTokenBudget),
  )
  expect_capacity_error(
    () => {
      @runtime_resolved.resolve(
        scheduler_fixture(),
        cache_fixture(),
        shape_fixture(),
        worker_fixture(max_sequence_tokens=4095),
      )
    },
    Invalid(WorkerSequenceTokens),
  )
}

///|
test "model and every worker row kind cover the per-step row ceiling" {
  expect_capacity_error(
    () => {
      @runtime_resolved.resolve(
        scheduler_fixture(),
        cache_fixture(),
        shape_fixture(max_batch_rows=7),
        worker_fixture(),
      )
    },
    Invalid(ModelBatchRows),
  )
  expect_capacity_error(
    () => {
      @runtime_resolved.resolve(
        scheduler_fixture(),
        cache_fixture(),
        shape_fixture(),
        worker_fixture(
          max_prefill_rows=7,
          max_decode_rows=7,
          max_plan_rows=7,
          max_completion_slots=7,
        ),
      )
    },
    Invalid(WorkerPlanRows),
  )
  expect_capacity_error(
    () => {
      @runtime_resolved.resolve(
        scheduler_fixture(),
        cache_fixture(),
        shape_fixture(),
        worker_fixture(max_prefill_rows=7),
      )
    },
    Invalid(WorkerPrefillRows),
  )
  expect_capacity_error(
    () => {
      @runtime_resolved.resolve(
        scheduler_fixture(),
        cache_fixture(),
        shape_fixture(),
        worker_fixture(max_decode_rows=7),
      )
    },
    Invalid(WorkerDecodeRows),
  )
}

///|
test "worker flattened page tables cover every maximum-sized row" {
  expect_capacity_error(
    () => {
      @runtime_resolved.resolve(
        scheduler_fixture(),
        cache_fixture(),
        shape_fixture(),
        worker_fixture(max_plan_pages=2047),
      )
    },
    Invalid(WorkerPlanPages),
  )
}

///|
test "worker token IDs cover the complete model vocabulary" {
  expect_capacity_error(
    () => {
      @runtime_resolved.resolve(
        scheduler_fixture(),
        cache_fixture(),
        shape_fixture(),
        worker_fixture(max_token_id=31998),
      )
    },
    Invalid(WorkerTokenIdRange),
  )
}

///|
test "canonical block-table materialization rejects excessive page cells" {
  expect_capacity_error(
    () => {
      @runtime_resolved.resolve(
        scheduler_fixture(
          step_token_budget=1,
          max_active_requests=65536,
          prefill_chunk_tokens=1,
          emergency_decode_page_reserve=1,
        ),
        cache_fixture(total_page_count=257, block_table_pages_per_request=257),
        shape_fixture(
          max_batch_rows=1,
          max_sequence_tokens=4096,
          vocabulary_size=128,
        ),
        worker_fixture(
          max_prefill_rows=1,
          max_decode_rows=1,
          max_plan_rows=1,
          max_plan_tokens=1,
          max_plan_pages=256,
          max_completion_slots=1,
          max_sequence_tokens=4096,
          max_token_id=127,
        ),
      )
    },
    Invalid(BlockTableLimits),
  )
}
