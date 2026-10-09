///|
fn prompt_limits(
  rows : Int,
  history : Int,
  vocabulary : Int,
) -> (@protocol.WorkerProtocolLimits, @inference.InferenceLimits) raise {
  (
    @protocol.WorkerProtocolLimits::new(
      max_prefill_rows=1,
      max_decode_rows=1,
      max_plan_rows=2,
      max_plan_tokens=rows,
      max_plan_pages=1,
      max_capabilities=1,
      max_completion_slots=2,
      max_sequence_tokens=history,
      max_token_id=vocabulary - 1,
    ),
    @inference.InferenceLimits::new(
      max_text_bytes=1048576,
      max_input_tokens=history,
      max_new_tokens=history,
      max_context_tokens=history,
      max_token_id=vocabulary - 1,
      max_stop_token_ids=1,
      max_stop_strings=1,
      max_stop_string_bytes=1,
      max_trace_bytes=1,
      max_cache_scope_bytes=1,
      max_decoded_delta_bytes=1,
      max_deadline_millis=300000UL,
      max_top_k=1,
      max_temperature=1.0,
    ),
  )
}
