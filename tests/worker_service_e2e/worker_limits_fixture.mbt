///|
fn worker_limits() -> @worker_protocol.WorkerProtocolLimits raise {
  @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=1,
    max_decode_rows=1,
    max_plan_rows=1,
    max_plan_tokens=1,
    max_plan_pages=2,
    max_capabilities=2,
    max_completion_slots=1,
    max_sequence_tokens=4,
    max_token_id=31,
  )
}
