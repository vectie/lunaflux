///|
/// Return or validate one exact admitted policy field.
pub fn InstancePolicyAdmission::digest(
  self : InstancePolicyAdmission,
) -> InstancePolicyDigest {
  self.digest
}

///|
/// Return or validate one exact admitted policy field.
pub fn InstancePolicyAdmission::runtime_descriptor_sha256(
  self : InstancePolicyAdmission,
) -> String {
  self.runtime_descriptor_sha256
}

///|
/// Return or validate one exact admitted policy field.
pub fn InstancePolicyAdmission::tokenizer(
  self : InstancePolicyAdmission,
) -> TokenizerPolicyClaims {
  self.tokenizer
}

///|
/// Return or validate one exact admitted policy field.
pub fn InstancePolicyAdmission::scheduler(
  self : InstancePolicyAdmission,
) -> SchedulerPolicyClaims {
  self.scheduler
}

///|
/// Return or validate one exact admitted policy field.
pub fn InstancePolicyAdmission::cache(
  self : InstancePolicyAdmission,
) -> CachePolicyClaims {
  self.cache
}

///|
/// Return or validate one exact admitted policy field.
pub fn InstancePolicyAdmission::service(
  self : InstancePolicyAdmission,
) -> ServicePolicyClaims {
  self.service
}

///|
/// Returns the inert, digest-bound external protocol claims.
pub fn InstancePolicyAdmission::external_protocol(
  self : InstancePolicyAdmission,
) -> ExternalProtocolPolicyClaims {
  self.external_protocol
}

///|
/// Returns the exact optional OpenAI construction claims. Only policy v3 with
/// OpenAI Responses intent can carry this inert value.
pub fn InstancePolicyAdmission::openai_service(
  self : InstancePolicyAdmission,
) -> OpenAiServicePolicyClaims? {
  self.openai_service
}

///|
/// Return or validate one exact admitted policy field.
pub fn InstancePolicyAdmission::worker_process(
  self : InstancePolicyAdmission,
) -> WorkerProcessPolicyClaims {
  self.worker_process
}

///|
/// Return or validate one exact admitted policy field.
pub fn InstancePolicyAdmission::restart(
  self : InstancePolicyAdmission,
) -> RestartPolicyClaims {
  self.restart
}

///|
/// Return or validate one exact admitted policy field.
pub fn InstancePolicyAdmission::transport(
  self : InstancePolicyAdmission,
) -> TransportPolicyClaims {
  self.transport
}

///|
/// Return or validate one exact admitted policy field.
pub fn InstancePolicyAdmission::preparation(
  self : InstancePolicyAdmission,
) -> PreparationPolicyClaims {
  self.preparation
}

///|
/// Return or validate one exact admitted policy field.
pub fn InstancePolicyAdmission::telemetry(
  self : InstancePolicyAdmission,
) -> TelemetryPolicyClaims {
  self.telemetry
}

///|
/// Return or validate one exact admitted policy field.
pub fn TokenizerPolicyClaims::locator(self : TokenizerPolicyClaims) -> String {
  self.locator
}

///|
/// Return or validate one exact admitted policy field.
pub fn TokenizerPolicyClaims::sha256(self : TokenizerPolicyClaims) -> String {
  self.sha256
}

///|
/// Return or validate one exact admitted policy field.
pub fn TokenizerPolicyClaims::max_file_bytes(
  self : TokenizerPolicyClaims,
) -> Int {
  self.max_file_bytes
}

///|
/// Return or validate one exact admitted policy field.
pub fn TokenizerPolicyClaims::max_json_depth(
  self : TokenizerPolicyClaims,
) -> Int {
  self.max_json_depth
}

///|
/// Return or validate one exact admitted policy field.
pub fn TokenizerPolicyClaims::max_input_bytes(
  self : TokenizerPolicyClaims,
) -> Int {
  self.max_input_bytes
}

///|
/// Return or validate one exact admitted policy field.
pub fn TokenizerPolicyClaims::max_output_tokens(
  self : TokenizerPolicyClaims,
) -> Int {
  self.max_output_tokens
}

///|
/// Return or validate one exact admitted policy field.
pub fn TokenizerPolicyClaims::max_decoded_bytes(
  self : TokenizerPolicyClaims,
) -> Int {
  self.max_decoded_bytes
}

///|
/// Return or validate one exact admitted policy field.
pub fn TokenizerPolicyClaims::max_vocab_entries(
  self : TokenizerPolicyClaims,
) -> Int {
  self.max_vocab_entries
}

///|
/// Return or validate one exact admitted policy field.
pub fn TokenizerPolicyClaims::max_merge_rules(
  self : TokenizerPolicyClaims,
) -> Int {
  self.max_merge_rules
}

///|
/// Return or validate one exact admitted policy field.
pub fn TokenizerPolicyClaims::max_token_bytes(
  self : TokenizerPolicyClaims,
) -> Int {
  self.max_token_bytes
}

///|
/// Return or validate one exact admitted policy field.
pub fn TokenizerPolicyClaims::max_special_tokens(
  self : TokenizerPolicyClaims,
) -> Int {
  self.max_special_tokens
}

///|
/// Return or validate one exact admitted policy field.
pub fn SchedulerPolicyClaims::step_token_budget(
  self : SchedulerPolicyClaims,
) -> Int {
  self.step_token_budget
}

///|
/// Return or validate one exact admitted policy field.
pub fn SchedulerPolicyClaims::max_active_requests(
  self : SchedulerPolicyClaims,
) -> Int {
  self.max_active_requests
}

///|
/// Return or validate one exact admitted policy field.
pub fn SchedulerPolicyClaims::max_waiting_requests(
  self : SchedulerPolicyClaims,
) -> Int {
  self.max_waiting_requests
}

///|
/// Return or validate one exact admitted policy field.
pub fn SchedulerPolicyClaims::prefill_chunk_tokens(
  self : SchedulerPolicyClaims,
) -> Int {
  self.prefill_chunk_tokens
}

///|
/// Return or validate one exact admitted policy field.
pub fn SchedulerPolicyClaims::emergency_decode_page_reserve(
  self : SchedulerPolicyClaims,
) -> Int {
  self.emergency_decode_page_reserve
}

///|
/// Return or validate one exact admitted policy field.
pub fn SchedulerPolicyClaims::waiting_age_threshold_steps(
  self : SchedulerPolicyClaims,
) -> Int {
  self.waiting_age_threshold_steps
}

///|
/// Return or validate one exact admitted policy field.
pub fn SchedulerPolicyClaims::output_event_capacity(
  self : SchedulerPolicyClaims,
) -> Int {
  self.output_event_capacity
}

///|
/// Return or validate one exact admitted policy field.
pub fn CachePolicyClaims::tokens_per_page(self : CachePolicyClaims) -> Int {
  self.tokens_per_page
}

///|
/// Return or validate one exact admitted policy field.
pub fn CachePolicyClaims::total_page_count(self : CachePolicyClaims) -> Int {
  self.total_page_count
}

///|
/// Return or validate one exact admitted policy field.
pub fn CachePolicyClaims::block_table_pages_per_request(
  self : CachePolicyClaims,
) -> Int {
  self.block_table_pages_per_request
}

///|
/// Return or validate one exact admitted policy field.
pub fn CachePolicyClaims::prefix_enabled(self : CachePolicyClaims) -> Bool {
  self.prefix_enabled
}

///|
/// Return or validate one exact admitted policy field.
pub fn CachePolicyClaims::max_prefix_entries(self : CachePolicyClaims) -> Int {
  self.max_prefix_entries
}

///|
/// Return or validate one exact admitted policy field.
pub fn CachePolicyClaims::max_prefix_nodes(self : CachePolicyClaims) -> Int {
  self.max_prefix_nodes
}

///|
/// Return or validate one exact admitted policy field.
pub fn CachePolicyClaims::max_prefix_tokens_per_entry(
  self : CachePolicyClaims,
) -> Int {
  self.max_prefix_tokens_per_entry
}

///|
/// Return or validate one exact admitted policy field.
pub fn CachePolicyClaims::max_prefix_pages(self : CachePolicyClaims) -> Int {
  self.max_prefix_pages
}

///|
/// Return or validate one exact admitted policy field.
pub fn CachePolicyClaims::max_prefix_scope_bytes(
  self : CachePolicyClaims,
) -> Int {
  self.max_prefix_scope_bytes
}

///|
/// Return or validate one exact admitted policy field.
pub fn CachePolicyClaims::max_active_references_per_page(
  self : CachePolicyClaims,
) -> Int {
  self.max_active_references_per_page
}

///|
/// Return or validate one exact admitted policy field.
pub fn CachePolicyClaims::max_cached_references_per_page(
  self : CachePolicyClaims,
) -> Int {
  self.max_cached_references_per_page
}

///|
/// Return or validate one exact admitted policy field.
pub fn CachePolicyClaims::layout_version(self : CachePolicyClaims) -> Int {
  self.layout_version
}
