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
  assert_eq(capacity.output_event_capacity(), 128)
  assert_eq(capacity.max_rows_per_step(), 8)
  assert_eq(capacity.maximum_generated_tokens_per_request(), 4095)
  assert_eq(capacity.required_recompute_history_cells(), 163800)
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
test "prefix startup binds only active tables and exact radix identity layout" {
  let capacity = @runtime_resolved.resolve(
    scheduler_fixture(max_active_requests=8, max_waiting_requests=32),
    prefix_cache_fixture(),
    shape_fixture(),
    worker_fixture(),
  )
  assert_true(capacity.prefix_enabled())
  assert_eq(capacity.block_table_limits().table_capacity(), 8)
  assert_eq(capacity.prefix_layout().version(), 7)
  assert_eq(capacity.prefix_layout().tokens_per_page(), 16)
  guard capacity.prefix_index_limits() is Some(limits) else {
    fail("enabled prefix limits were absent")
  }
  assert_eq(limits.entry_capacity(), 64)
  assert_eq(limits.node_capacity(), 1024)
  assert_eq(limits.token_capacity(), 1024)
  assert_eq(limits.max_tokens_per_entry(), 256)
  assert_eq(limits.page_capacity(), 256)
  assert_eq(limits.max_scope_bytes(), 64)
  assert_eq(limits.max_active_references_per_entry(), 8)
}

///|
test "large waiting capacity retains logical storage without physical owners" {
  let capacity = @runtime_resolved.resolve(
    scheduler_fixture(
      step_token_budget=2,
      max_active_requests=2,
      max_waiting_requests=1024,
      prefill_chunk_tokens=2,
      output_event_capacity=2,
    ),
    prefix_cache_fixture(max_active_references_per_page=2),
    shape_fixture(max_batch_rows=2, max_sequence_tokens=64),
    worker_fixture(
      max_prefill_rows=2,
      max_decode_rows=2,
      max_plan_rows=2,
      max_plan_tokens=2,
      max_plan_pages=8,
      max_completion_slots=2,
      max_sequence_tokens=64,
    ),
  )
  assert_eq(capacity.max_active_requests(), 2)
  assert_eq(capacity.max_waiting_requests(), 1024)
  assert_eq(capacity.total_request_slots(), 1026)
  assert_eq(capacity.required_recompute_history_cells(), 64638)
  assert_eq(capacity.block_table_limits().table_capacity(), 2)
  guard capacity.prefix_index_limits() is Some(limits) else {
    fail("enabled prefix limits were absent")
  }
  assert_eq(limits.max_active_references_per_entry(), 2)
  assert_eq(
    capacity.page_allocator_limits().max_active_references_per_page(),
    2,
  )
}

///|
test "prefix aggregate scope arena is rejected before scheduler allocation" {
  expect_capacity_error(
    () => {
      @runtime_resolved.resolve(
        scheduler_fixture(),
        prefix_cache_fixture(
          max_prefix_entries=65536,
          max_prefix_nodes=65536,
          max_prefix_scope_bytes=257,
        ),
        shape_fixture(),
        worker_fixture(),
      )
    },
    Invalid(PrefixIndexLimits),
  )
}

///|
test "prefix startup rejects unusable page-aligned publication envelopes" {
  expect_capacity_error(
    () => {
      @runtime_resolved.resolve(
        scheduler_fixture(),
        prefix_cache_fixture(max_prefix_tokens_per_entry=15),
        shape_fixture(),
        worker_fixture(),
      )
    },
    Invalid(PrefixIndexLimits),
  )
  expect_capacity_error(
    () => {
      @runtime_resolved.resolve(
        scheduler_fixture(),
        prefix_cache_fixture(
          max_prefix_tokens_per_entry=256,
          max_prefix_pages=15,
        ),
        shape_fixture(),
        worker_fixture(),
      )
    },
    Invalid(PrefixIndexLimits),
  )
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
  assert_eq(decisions.length(), 8)
  assert_eq(decisions[0].subject(), ContextPagesPerRequest)
  assert_eq(decisions[0].reason(), CeilingDivision)
  assert_eq(decisions[2].subject(), MaxRowsPerStep)
  assert_eq(decisions[2].reason(), MinimumOfActiveAndTokenBudget)
  assert_eq(decisions[4].subject(), RecomputeHistoryCells)
  assert_eq(decisions[4].reason(), RequestSlotsTimesMaximumGeneratedTokens)
  assert_eq(decisions[5].reason(), FixedDoubleBuffer)
  assert_eq(decisions[6].reason(), OwnerMaximumGenerationDefault)
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
test "physical pages bound flattened tables without active sharing" {
  let scheduler = scheduler_fixture(
    step_token_budget=256,
    max_active_requests=32,
    max_waiting_requests=32,
    prefill_chunk_tokens=256,
    output_event_capacity=16384,
  )
  let cache = cache_fixture(
    tokens_per_page=8,
    total_page_count=8192,
    block_table_pages_per_request=5120,
    max_active_references_per_page=1,
  )
  let shapes = shape_fixture(
    max_batch_rows=32,
    max_sequence_tokens=40960,
    vocabulary_size=151936,
  )
  let worker = worker_fixture(
    max_prefill_rows=32,
    max_decode_rows=32,
    max_plan_rows=32,
    max_plan_tokens=256,
    max_plan_pages=8192,
    max_completion_slots=32,
    max_sequence_tokens=40960,
    max_token_id=151935,
  )
  let capacity = @runtime_resolved.resolve(scheduler, cache, shapes, worker)
  assert_eq(capacity.required_worker_page_table_cells(), 8192)
  expect_capacity_error(
    () => {
      @runtime_resolved.resolve(
        scheduler,
        cache,
        shapes,
        worker_fixture(
          max_prefill_rows=32,
          max_decode_rows=32,
          max_plan_rows=32,
          max_plan_tokens=256,
          max_plan_pages=8191,
          max_completion_slots=32,
          max_sequence_tokens=40960,
          max_token_id=151935,
        ),
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

///|
test "recompute history rejects checked slot times token overflow" {
  expect_capacity_error(
    () => {
      @runtime_resolved.resolve(
        scheduler_fixture(
          step_token_budget=1,
          max_active_requests=65536,
          max_waiting_requests=983040,
          prefill_chunk_tokens=1,
          emergency_decode_page_reserve=1,
          output_event_capacity=1,
        ),
        cache_fixture(total_page_count=257, block_table_pages_per_request=256),
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
    Invalid(RecomputeHistoryCapacity),
  )
}
