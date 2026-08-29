///|
/// Build a typed operator report from pure joined instance admission.
pub fn build_instance(
  admission : @instance_admission.RuntimeInstanceAdmission,
) -> RuntimeInstanceReport raise RuntimeReportError {
  let runtime = build(admission.runtime_descriptor())
  let capacity = admission.capacity()
  {
    runtime,
    tokenizer_vocabulary_size: admission.tokenizer().vocab_size(),
    scheduler_total_request_slots: capacity.total_request_slots(),
    scheduler_max_active_requests: capacity.max_active_requests(),
    scheduler_max_waiting_requests: capacity.max_waiting_requests(),
    scheduler_max_rows_per_step: capacity.max_rows_per_step(),
    scheduler_prefix_enabled: capacity.prefix_enabled(),
    service_max_request_bytes: admission.service_config().max_request_bytes(),
    preparation_lane_count: admission.preparation_lane_count(),
    telemetry_capacity: admission.telemetry_capacity(),
  }
}
