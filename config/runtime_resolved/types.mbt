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
  RecomputeHistoryCapacity
  WorkerTokenIdRange
  OutputEventCapacity
  ModelBatchRows
  PageAllocatorLimits
  BlockTableLimits
  PrefixIndexLimits
  GraphMemoryEvidence
  GraphMemoryCeiling
  DeviceMemoryUpperBound
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
  RecomputeHistoryCells
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
  MinimumOfRowAndPhysicalReferenceBounds
  RequestSlotsTimesMaximumGeneratedTokens
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
  priv maximum_generated_tokens_per_request : Int
  priv required_recompute_history_cells : Int
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
  priv prefix_index_limits : @radix.PrefixIndexLimits?
  priv prefix_layout : @radix.PrefixLayout
  priv decisions : ReadOnlyArray[RuntimeCapacityDecision]
} derive(Debug, Eq)

///|
/// Stable availability of authenticated CUDA-graph memory accounting.
pub(all) enum RuntimeGraphMemoryAvailability {
  GraphAccountingAbsent
  GraphAccountingUnsupported
  GraphAccountingBounded
} derive(Debug, Eq)

///|
/// Opaque startup resolution of arena reservations and an authenticated
/// CUDA-graph declared upper bound.
///
/// The arena total contains weight, activation/workspace, and persistent KV
/// exactly once. The graph-aware startup upper bound exists only when capture
/// metadata and a positive v2 descriptor ceiling were both admitted. An
/// eager-only v1 descriptor never projects an inert future graph bound.
pub struct ResolvedGraphMemoryCapacity {
  priv availability : RuntimeGraphMemoryAvailability
  priv arena_reserved_total_bytes : Int64
  priv declared_upper_bound_bytes : Int64?
  priv declared_alignment_bytes : Int64?
  priv descriptor_ceiling_bytes : Int64?
  priv device_startup_upper_bound_bytes : Int64?
} derive(Debug, Eq)
