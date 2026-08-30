///|
const REQUIRED_PLAN_BUFFER_COUNT : Int = 2

///|
const MAX_SAFE_INT : Int = 2_147_483_647

///|
fn checked_add_nonnegative(left : Int, right : Int) -> Int? {
  if left < 0 || right < 0 || left > MAX_SAFE_INT - right {
    None
  } else {
    Some(left + right)
  }
}

///|
fn checked_multiply_nonnegative(left : Int, right : Int) -> Int? {
  if left < 0 || right < 0 {
    None
  } else if left == 0 || right == 0 {
    Some(0)
  } else if left > MAX_SAFE_INT / right {
    None
  } else {
    Some(left * right)
  }
}

///|
fn ceiling_divide_positive(numerator : Int, denominator : Int) -> Int? {
  if numerator <= 0 || denominator <= 0 {
    None
  } else {
    Some((numerator - 1) / denominator + 1)
  }
}

///|
fn require_at_least(
  available : Int,
  required : Int,
  rule : RuntimeCapacityRule,
) -> Unit raise RuntimeCapacityError {
  if available < required {
    raise Invalid(rule)
  }
}

///|
fn materialize_page_allocator_limits(
  cache : @cache.CacheConfig,
) -> @page_allocator.PageAllocatorLimits raise RuntimeCapacityError {
  @page_allocator.PageAllocatorLimits::new(
    page_capacity=cache.total_page_count(),
    max_run_pages=cache.block_table_pages_per_request(),
    max_active_references_per_page=cache.max_active_references_per_page(),
    max_cached_references_per_page=cache.max_cached_references_per_page(),
  ) catch {
    _ => raise Invalid(PageAllocatorLimits)
  }
}

///|
fn materialize_block_table_limits(
  cache : @cache.CacheConfig,
  table_capacity : Int,
) -> @block_table.BlockTableLimits raise RuntimeCapacityError {
  @block_table.BlockTableLimits::new(
    table_capacity~,
    max_pages_per_table=cache.block_table_pages_per_request(),
    tokens_per_page=cache.tokens_per_page(),
  ) catch {
    _ => raise Invalid(BlockTableLimits)
  }
}

///|
/// Resolves exact cross-component capacities before runtime owners allocate.
///
/// `max_rows_per_step` is `min(max_active_requests, step_token_budget)` because
/// every submitted row consumes at least one token. A legal step can consist
/// entirely of decode rows or one-token prefill rows, so model batch rows and
/// every worker row/completion envelope must independently support that full
/// ceiling. Active concurrency may exceed this per-step row ceiling.
///
/// Worker page storage is bounded by both the row/context envelope and the
/// number of active references that the physical page pool can own. This
/// avoids requiring an impossible all-rows-full envelope when the KV pool is
/// smaller, while retaining the larger bound when active page sharing permits
/// duplicate flattened references.
pub fn resolve(
  scheduler : @scheduler.SchedulerConfig,
  cache : @cache.CacheConfig,
  shapes : @plan.ShapeConstraints,
  worker : @worker_protocol.WorkerProtocolLimits,
) -> ResolvedRuntimeCapacity raise RuntimeCapacityError {
  let tokens_per_page = cache.tokens_per_page()
  let total_page_count = cache.total_page_count()
  let block_table_pages_per_request = cache.block_table_pages_per_request()
  let model_max_sequence_tokens = shapes.max_sequence_tokens()
  let context_pages_per_request = match
    ceiling_divide_positive(model_max_sequence_tokens, tokens_per_page) {
    Some(value) => value
    None => raise Invalid(ContextFitsBlockTable)
  }
  if scheduler.emergency_decode_page_reserve() > total_page_count {
    raise Invalid(EmergencyReserveWithinPhysicalPages)
  }
  if context_pages_per_request > block_table_pages_per_request {
    raise Invalid(ContextFitsBlockTable)
  }
  let max_active_requests = scheduler.max_active_requests()
  let max_waiting_requests = scheduler.max_waiting_requests()
  let total_request_slots = match
    checked_add_nonnegative(max_active_requests, max_waiting_requests) {
    Some(value) => value
    None => raise Invalid(TotalRequestSlots)
  }
  let step_token_budget = scheduler.step_token_budget()
  let max_rows_per_step = if max_active_requests < step_token_budget {
    max_active_requests
  } else {
    step_token_budget
  }
  let output_event_capacity = scheduler.output_event_capacity()
  require_at_least(
    output_event_capacity,
    max_rows_per_step,
    OutputEventCapacity,
  )
  require_at_least(shapes.max_batch_rows(), max_rows_per_step, ModelBatchRows)
  require_at_least(
    worker.max_plan_tokens(),
    step_token_budget,
    WorkerTokenBudget,
  )
  require_at_least(worker.max_plan_rows(), max_rows_per_step, WorkerPlanRows)
  require_at_least(
    worker.max_prefill_rows(),
    max_rows_per_step,
    WorkerPrefillRows,
  )
  require_at_least(
    worker.max_decode_rows(),
    max_rows_per_step,
    WorkerDecodeRows,
  )
  require_at_least(
    worker.max_completion_slots(),
    max_rows_per_step,
    WorkerCompletionSlots,
  )
  require_at_least(
    worker.max_sequence_tokens(),
    model_max_sequence_tokens,
    WorkerSequenceTokens,
  )
  require_at_least(
    worker.max_token_id(),
    shapes.vocabulary_size() - 1,
    WorkerTokenIdRange,
  )
  let row_context_page_table_cells = match
    checked_multiply_nonnegative(max_rows_per_step, context_pages_per_request) {
    Some(value) => value
    None => raise Invalid(WorkerPlanPages)
  }
  let physical_reference_page_table_cells = match
    checked_multiply_nonnegative(
      total_page_count,
      cache.max_active_references_per_page(),
    ) {
    Some(value) => value
    None => MAX_SAFE_INT
  }
  let required_worker_page_table_cells = Int::min(
    row_context_page_table_cells, physical_reference_page_table_cells,
  )
  require_at_least(
    worker.max_plan_pages(),
    required_worker_page_table_cells,
    WorkerPlanPages,
  )
  let maximum_generated_tokens_per_request = model_max_sequence_tokens - 1
  let required_recompute_history_cells = match
    checked_multiply_nonnegative(
      total_request_slots, maximum_generated_tokens_per_request,
    ) {
    Some(value) => value
    None => raise Invalid(RecomputeHistoryCapacity)
  }
  let page_allocator_limits = materialize_page_allocator_limits(cache)
  let block_table_capacity = scheduler.max_active_requests()
  let block_table_limits = materialize_block_table_limits(
    cache, block_table_capacity,
  )
  let prefix_layout = @radix.PrefixLayout::new(
    version=cache.layout_version(),
    tokens_per_page=cache.tokens_per_page(),
  ) catch {
    _ => raise Invalid(PrefixIndexLimits)
  }
  let prefix_index_limits = if cache.prefix_enabled() {
    if cache.max_active_references_per_page() < scheduler.max_active_requests() {
      raise Invalid(PrefixIndexLimits)
    }
    if cache.max_prefix_tokens_per_entry() < tokens_per_page ||
      cache.max_prefix_pages() <
      cache.max_prefix_tokens_per_entry() / tokens_per_page {
      raise Invalid(PrefixIndexLimits)
    }
    Some(
      @radix.PrefixIndexLimits::new(
        entry_capacity=cache.max_prefix_entries(),
        node_capacity=cache.max_prefix_nodes(),
        token_capacity=cache.max_prefix_nodes(),
        max_tokens_per_entry=cache.max_prefix_tokens_per_entry(),
        page_capacity=cache.max_prefix_pages(),
        max_scope_bytes=cache.max_prefix_scope_bytes(),
        max_active_references_per_entry=scheduler.max_active_requests(),
      ) catch {
        _ => raise Invalid(PrefixIndexLimits)
      },
    )
  } else {
    None
  }
  let decisions : Array[RuntimeCapacityDecision] = [
    { subject: ContextPagesPerRequest, reason: CeilingDivision },
    { subject: TotalRequestSlots, reason: CombinedConfiguredCapacity },
    { subject: MaxRowsPerStep, reason: MinimumOfActiveAndTokenBudget },
    {
      subject: WorkerPageTableCells,
      reason: MinimumOfRowAndPhysicalReferenceBounds,
    },
    {
      subject: RecomputeHistoryCells,
      reason: RequestSlotsTimesMaximumGeneratedTokens,
    },
    { subject: PlanBufferCount, reason: FixedDoubleBuffer },
    { subject: PageGenerationLimit, reason: OwnerMaximumGenerationDefault },
    {
      subject: BlockTableGenerationLimit,
      reason: OwnerMaximumGenerationDefault,
    },
  ]
  {
    tokens_per_page,
    total_page_count,
    context_pages_per_request,
    block_table_pages_per_request,
    max_active_requests,
    max_waiting_requests,
    total_request_slots,
    step_token_budget,
    prefill_chunk_tokens: scheduler.prefill_chunk_tokens(),
    emergency_decode_page_reserve: scheduler.emergency_decode_page_reserve(),
    waiting_age_threshold_steps: scheduler.waiting_age_threshold_steps(),
    output_event_capacity,
    max_rows_per_step,
    required_worker_page_table_cells,
    maximum_generated_tokens_per_request,
    required_recompute_history_cells,
    model_max_batch_rows: shapes.max_batch_rows(),
    model_max_sequence_tokens,
    worker_max_prefill_rows: worker.max_prefill_rows(),
    worker_max_decode_rows: worker.max_decode_rows(),
    worker_max_plan_rows: worker.max_plan_rows(),
    worker_max_plan_tokens: worker.max_plan_tokens(),
    worker_max_plan_pages: worker.max_plan_pages(),
    worker_max_capabilities: worker.max_capabilities(),
    worker_max_completion_slots: worker.max_completion_slots(),
    worker_max_sequence_tokens: worker.max_sequence_tokens(),
    worker_max_token_id: worker.max_token_id(),
    page_allocator_limits,
    block_table_limits,
    prefix_index_limits,
    prefix_layout,
    decisions: ReadOnlyArray::from_array(decisions),
  }
}
