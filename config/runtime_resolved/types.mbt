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
  OutputEventCapacity
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
  priv tokens_per_page : Int
  priv total_page_count : Int
  priv context_pages_per_request : Int
  priv block_table_pages_per_request : Int
  priv max_active_requests : Int
  priv max_waiting_requests : Int
  priv total_request_slots : Int
  priv step_token_budget : Int
  priv prefill_chunk_tokens : Int
  priv emergency_decode_page_reserve : Int
  priv waiting_age_threshold_steps : Int
  priv output_event_capacity : Int
  priv max_rows_per_step : Int
  priv required_worker_page_table_cells : Int
  priv model_max_batch_rows : Int
  priv model_max_sequence_tokens : Int
  priv worker_max_prefill_rows : Int
  priv worker_max_decode_rows : Int
  priv worker_max_plan_rows : Int
  priv worker_max_plan_tokens : Int
  priv worker_max_plan_pages : Int
  priv worker_max_capabilities : Int
  priv worker_max_completion_slots : Int
  priv worker_max_sequence_tokens : Int
  priv worker_max_token_id : Int
  priv page_allocator_limits : @page_allocator.PageAllocatorLimits
  priv block_table_limits : @block_table.BlockTableLimits
  priv decisions : ReadOnlyArray[RuntimeCapacityDecision]
} derive(Debug, Eq)
