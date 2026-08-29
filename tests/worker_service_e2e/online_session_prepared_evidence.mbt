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
fn assert_stale_preparation_claim(
  claim : @request_admission.LunaPreparedRequestClaim,
) -> Unit raise {
  try ignore(claim.scheduler_request()) catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => raise Failure("unexpected stale Luna claim read error")
  } noraise {
    _ => raise Failure("released Luna claim retained scheduler authority")
  }
  try claim.release() catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => raise Failure("unexpected stale Luna claim release error")
  } noraise {
    _ => raise Failure("released Luna claim returned a reused lane twice")
  }
  let destination = FixedArray::make(8, b'!')
  assert_false(
    claim
    .push_token_into_status(1, destination, destination_offset=0)
    .is_ready(),
  )
  assert_false(
    claim.finish_into_status(destination, destination_offset=0).is_ready(),
  )
}

///|
fn run_preparation_pool_alias_evidence() -> Unit raise {
  let tokenizer = session_tokenizer()
  let pool = prepare_foundation_pool(tokenizer, 1)
  let first_work = submit_foundation_pool_work(
    pool,
    session_received(Streaming, request_id=940UL),
  )
  let first_prepared = finish_foundation_pool_work(pool, first_work)
  let first_prepared_alias = first_prepared
  let saturated_received = session_received(Streaming, request_id=941UL)
  assert_eq(
    pool.try_submit(saturated_received).kind(),
    LunaRequestPreparationSaturated,
  )
  let first_claim = first_prepared.take_claim()
  assert_eq(first_claim.scheduler_request().request_id().value(), 940UL)
  first_claim.release()
  assert_eq(pool.free_lane_count(), 1)

  // The saturated receipt was not consumed. Its admission reuses the only
  // lane, advancing the exact storage generation and staling every old alias.
  let second_work = submit_foundation_pool_work(pool, saturated_received)
  let second_prepared = finish_foundation_pool_work(pool, second_work)
  assert_pool_work_stale(first_work)
  assert_prepared_shell_revoked(first_prepared_alias)
  assert_stale_preparation_claim(first_claim)
  second_prepared.discard()
  assert_eq(pool.free_lane_count(), 1)

  let draining_received = session_received(Streaming, request_id=942UL)
  pool.begin_drain()
  assert_eq(
    pool.try_submit(draining_received).kind(),
    LunaRequestPreparationDraining,
  )
  assert_true(pool.is_drained())
  let retry_pool = prepare_foundation_pool(tokenizer, 1)
  let retry_prepared = prepare_foundation_pool_request(
    retry_pool, draining_received,
  )
  retry_prepared.discard()
  assert_eq(retry_pool.free_lane_count(), 1)
}

///|
fn run_online_session_prepared_evidence(executable : Bytes) -> Unit raise {
  run_preparation_pool_alias_evidence()
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
  drain_foundation_instance(draining_instance)
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
