///|
/// Cross-component startup relations checked by runtime capacity resolution.
pub(all) enum RuntimeCapacityRule {
  EmergencyReserveWithinPhysicalPages
  ContextFitsBlockTable
  TotalRequestSlots
  WorkerTokenBudget
  WorkerPlanRows
  WorkerPrefillRows
  WorkerDecodeRows
  WorkerCompletionSlots
  WorkerSequenceTokens
  WorkerPlanPages
  WorkerTokenIdRange
  ModelBatchRows
  PageAllocatorLimits
  BlockTableLimits
} derive(Debug, Eq)

///|
/// Payload-safe startup failure. The rule names only bounded configuration
/// categories; it contains no model path, request data, or device details.
pub(all) suberror RuntimeCapacityError {
  Invalid(RuntimeCapacityRule)
} derive(Debug, Eq)

///|
/// Stable subjects for derived startup-capacity explanations.
pub(all) enum RuntimeCapacitySubject {
  ContextPagesPerRequest
  TotalRequestSlots
  MaxRowsPerStep
  WorkerPageTableCells
  PlanBufferCount
  PageGenerationLimit
  BlockTableGenerationLimit
} derive(Debug, Eq)

///|
/// Stable reasons for deterministic startup-capacity decisions.
pub(all) enum RuntimeCapacityReason {
  CeilingDivision
  CombinedConfiguredCapacity
  MinimumOfActiveAndTokenBudget
  RowsTimesContextPages
  FixedDoubleBuffer
  OwnerMaximumGenerationDefault
} derive(Debug, Eq)

///|
/// One bounded explanation for a derived startup capacity.
pub struct RuntimeCapacityDecision {
  subject : RuntimeCapacitySubject
  reason : RuntimeCapacityReason
} derive(Debug, Eq)

///|
/// Immutable startup-only materialization of scheduler, cache, model-shape,
/// worker-protocol, and host KV metadata capacity relations.
///
/// It deliberately contains no device KV byte count, kernel support claim, or
/// live mutable runtime state.
pub struct ResolvedRuntimeCapacity {
  tokens_per_page : Int
  total_page_count : Int
  context_pages_per_request : Int
  block_table_pages_per_request : Int
  max_active_requests : Int
  max_waiting_requests : Int
  total_request_slots : Int
  step_token_budget : Int
  prefill_chunk_tokens : Int
  emergency_decode_page_reserve : Int
  waiting_age_threshold_steps : Int
  max_rows_per_step : Int
  required_worker_page_table_cells : Int
  model_max_batch_rows : Int
  model_max_sequence_tokens : Int
  worker_max_prefill_rows : Int
  worker_max_decode_rows : Int
  worker_max_plan_rows : Int
  worker_max_plan_tokens : Int
  worker_max_plan_pages : Int
  worker_max_capabilities : Int
  worker_max_completion_slots : Int
  worker_max_sequence_tokens : Int
  worker_max_token_id : Int
  page_allocator_limits : @page_allocator.PageAllocatorLimits
  block_table_limits : @block_table.BlockTableLimits
  decisions : ReadOnlyArray[RuntimeCapacityDecision]
} derive(Debug, Eq)
