///|
pub fn RuntimeCapacityDecision::subject(
  self : RuntimeCapacityDecision,
) -> RuntimeCapacitySubject {
  self.subject
}

///|
pub fn RuntimeCapacityDecision::reason(
  self : RuntimeCapacityDecision,
) -> RuntimeCapacityReason {
  self.reason
}

///|
pub fn ResolvedRuntimeCapacity::tokens_per_page(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.tokens_per_page
}

///|
pub fn ResolvedRuntimeCapacity::total_page_count(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.total_page_count
}

///|
pub fn ResolvedRuntimeCapacity::context_pages_per_request(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.context_pages_per_request
}

///|
pub fn ResolvedRuntimeCapacity::block_table_pages_per_request(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.block_table_pages_per_request
}

///|
pub fn ResolvedRuntimeCapacity::max_active_requests(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.max_active_requests
}

///|
pub fn ResolvedRuntimeCapacity::max_waiting_requests(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.max_waiting_requests
}

///|
pub fn ResolvedRuntimeCapacity::total_request_slots(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.total_request_slots
}

///|
pub fn ResolvedRuntimeCapacity::step_token_budget(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.step_token_budget
}

///|
pub fn ResolvedRuntimeCapacity::prefill_chunk_tokens(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.prefill_chunk_tokens
}

///|
pub fn ResolvedRuntimeCapacity::emergency_decode_page_reserve(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.emergency_decode_page_reserve
}

///|
pub fn ResolvedRuntimeCapacity::waiting_age_threshold_steps(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.waiting_age_threshold_steps
}

///|
pub fn ResolvedRuntimeCapacity::max_rows_per_step(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.max_rows_per_step
}

///|
pub fn ResolvedRuntimeCapacity::required_worker_page_table_cells(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.required_worker_page_table_cells
}

///|
pub fn ResolvedRuntimeCapacity::plan_buffer_count(
  _self : ResolvedRuntimeCapacity,
) -> Int {
  REQUIRED_PLAN_BUFFER_COUNT
}

///|
pub fn ResolvedRuntimeCapacity::model_max_batch_rows(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.model_max_batch_rows
}

///|
pub fn ResolvedRuntimeCapacity::model_max_sequence_tokens(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.model_max_sequence_tokens
}

///|
pub fn ResolvedRuntimeCapacity::worker_max_prefill_rows(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.worker_max_prefill_rows
}

///|
pub fn ResolvedRuntimeCapacity::worker_max_decode_rows(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.worker_max_decode_rows
}

///|
pub fn ResolvedRuntimeCapacity::worker_max_plan_rows(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.worker_max_plan_rows
}

///|
pub fn ResolvedRuntimeCapacity::worker_max_plan_tokens(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.worker_max_plan_tokens
}

///|
pub fn ResolvedRuntimeCapacity::worker_max_plan_pages(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.worker_max_plan_pages
}

///|
pub fn ResolvedRuntimeCapacity::worker_max_capabilities(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.worker_max_capabilities
}

///|
pub fn ResolvedRuntimeCapacity::worker_max_completion_slots(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.worker_max_completion_slots
}

///|
pub fn ResolvedRuntimeCapacity::worker_max_sequence_tokens(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.worker_max_sequence_tokens
}

///|
pub fn ResolvedRuntimeCapacity::worker_max_token_id(
  self : ResolvedRuntimeCapacity,
) -> Int {
  self.worker_max_token_id
}

///|
/// Canonical fixed-page metadata limits. The generation ceiling is the owner
/// package's maximum default, so exhausted identities retire instead of wrap.
pub fn ResolvedRuntimeCapacity::page_allocator_limits(
  self : ResolvedRuntimeCapacity,
) -> @page_allocator.PageAllocatorLimits {
  self.page_allocator_limits
}

///|
/// Canonical active-request block-table limits. Waiting requests own no table.
/// The generation ceiling is the owner package's maximum default.
pub fn ResolvedRuntimeCapacity::block_table_limits(
  self : ResolvedRuntimeCapacity,
) -> @block_table.BlockTableLimits {
  self.block_table_limits
}

///|
pub fn ResolvedRuntimeCapacity::decisions(
  self : ResolvedRuntimeCapacity,
) -> ReadOnlyArray[RuntimeCapacityDecision] {
  self.decisions
}
