///|
fn alternate_inference_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=63,
    max_input_tokens=4,
    max_new_tokens=2,
    max_context_tokens=4,
    max_token_id=31,
    max_stop_token_ids=1,
    max_stop_strings=1,
    max_stop_string_bytes=2,
    max_trace_bytes=8,
    max_cache_scope_bytes=8,
    max_decoded_delta_bytes=8,
    max_deadline_millis=1000UL,
    max_top_k=32,
    max_temperature=4.0,
  )
}

///|
fn run_online_session_prepared_evidence(executable : Bytes) -> Unit raise {
  let tokenizer = session_tokenizer()
  let instance = prepare_foundation_instance(executable, tokenizer.digest())
  let foreign_tokenizer = invalid_output_session_tokenizer()
  let foreign_prepared = prepare_foundation_request(
    foreign_tokenizer,
    session_received(Streaming, request_id=918UL),
  )
  try instance.begin(foreign_prepared) catch {
    LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure) => ()
    _ => raise Failure("unexpected foreign prepared-tokenizer error")
  } noraise {
    _ => raise Failure("foreign prepared tokenizer entered the instance")
  }
  let foreign_instance = prepare_foundation_instance(
    executable,
    foreign_tokenizer.digest(),
  )
  let foreign_session = begin_foundation_session(
    foreign_instance, foreign_prepared,
  )
  foreign_session.ack_event()
  foreign_session.begin_abort()
  progress_foundation_to_idle(foreign_session)
  drain_foundation_instance(foreign_instance)

  let envelope_prepared = @request_admission.prepare_luna_request(
    session_received(Streaming, request_id=919UL),
    tokenizer,
    tokenizer.digest(),
    identity(),
    alternate_inference_limits(),
    @monotonic_clock.MonotonicClock::system(),
  )
  try instance.begin(envelope_prepared) catch {
    LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineInferenceEnvelope)) =>
      ()
    _ => raise Failure("unexpected prepared-envelope mismatch error")
  } noraise {
    _ => raise Failure("mismatched prepared envelope entered the instance")
  }
  assert_eq(
    envelope_prepared.take_claim().scheduler_request().request_id().value(),
    919UL,
  )
  drain_foundation_instance(instance)

  let drain_retry_prepared = prepare_foundation_request(
    tokenizer,
    session_received(Streaming, request_id=917UL),
  )
  let draining_instance = prepare_foundation_instance(
    executable,
    tokenizer.digest(),
  )
  draining_instance.begin_drain()
  assert_eq(
    draining_instance.begin(drain_retry_prepared).kind(),
    LunaOnlineInstanceDraining,
  )
  assert_eq(
    draining_instance.progress_shutdown(),
    LunaOnlineInstanceShutdownClosed,
  )
  let drain_retry_instance = prepare_foundation_instance(
    executable,
    tokenizer.digest(),
  )
  let drain_retry = begin_foundation_session(
    drain_retry_instance, drain_retry_prepared,
  )
  drain_retry.ack_event()
  drain_retry.begin_abort()
  progress_foundation_to_idle(drain_retry)
  drain_foundation_instance(drain_retry_instance)
}
