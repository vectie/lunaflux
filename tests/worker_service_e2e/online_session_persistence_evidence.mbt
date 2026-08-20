///|
fn assert_stale_online_ticket(
  instance : @online_session.LunaOnlineInstance,
  stale : @online_session.LunaOnlineRequestTicket,
) -> Unit raise {
  try ignore(instance.take_event(stale)) catch {
    LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineRequestEpoch)) => ()
    _ => raise Failure("unexpected stale online request ticket error")
  } noraise {
    _ => raise Failure("stale online request ticket retained authority")
  }
}

///|
fn assert_prepared_shell_revoked(
  retained : @request_admission.LunaPreparedRequest,
) -> Unit raise {
  try ignore(retained.input_token_count()) catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => raise Failure("unexpected revoked Luna preparation error")
  } noraise {
    _ => raise Failure("retained Luna preparation alias kept claim authority")
  }
}

///|
fn assert_pool_work_stale(
  work : @request_admission.LunaRequestPreparationWork,
) -> Unit raise {
  try ignore(work.state()) catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => raise Failure("unexpected stale Luna preparation-work error")
  } noraise {
    _ => raise Failure("retained Luna preparation work crossed generations")
  }
}

///|
fn finish_persistent_natural(session : FoundationSession) -> Unit raise {
  session.ack_event()
  progress_foundation_to_event(session)
  match foundation_event(session) {
    Token(value) => {
      assert_eq(value.token_id(), 0)
      assert_eq(value.position(), 0)
    }
    _ => raise Failure("first persistent request did not publish Token(0)")
  }
  session.ack_event()
  progress_foundation_to_event(session)
  match foundation_event(session) {
    Usage(value) => assert_eq(value.output_tokens(), 1)
    _ => raise Failure("first persistent request did not publish Usage")
  }
  session.ack_event()
  match foundation_event(session) {
    Completed(value) => assert_eq(value.finish_reason(), TokenLimit)
    _ => raise Failure("first persistent request did not complete naturally")
  }
  session.ack_event()
  progress_foundation_to_idle(session)
}

///|
fn finish_persistent_cancelled(session : FoundationSession) -> Unit raise {
  session.ack_event()
  progress_foundation_to_event(session)
  match foundation_event(session) {
    Token(value) => {
      assert_eq(value.token_id(), 0)
      assert_eq(value.position(), 0)
    }
    _ => raise Failure("second persistent request did not reset Token position")
  }
  session.request_cancel()
  session.ack_event()
  progress_foundation_to_event(session)
  match foundation_event(session) {
    Usage(value) => assert_eq(value.output_tokens(), 1)
    _ => raise Failure("second persistent request did not publish Usage")
  }
  session.ack_event()
  match foundation_event(session) {
    Completed(value) => {
      assert_eq(value.finish_reason(), Cancelled)
      assert_eq(value.final_usage().output_tokens(), 1)
    }
    _ => raise Failure("second persistent request did not cancel")
  }
  session.ack_event()
  progress_foundation_to_idle(session)
}

///|
fn run_online_session_persistence_evidence(
  executable : Bytes,
  persistent_executable : Bytes,
) -> Unit raise {
  let tokenizer = session_tokenizer()
  let pool = prepare_foundation_pool(tokenizer, 2)
  let instance = prepare_foundation_instance(
    persistent_executable,
    tokenizer.digest(),
  )
  let first_work = submit_foundation_pool_work(
    pool,
    session_received(
      Streaming,
      request_id=950UL,
      max_new_tokens=1,
      deadline_millis=1000UL,
    ),
  )
  let first_prepared = finish_foundation_pool_work(pool, first_work)
  let retained_first = first_prepared
  let second_work = submit_foundation_pool_work(
    pool,
    session_received(
      Streaming,
      request_id=951UL,
      max_new_tokens=2,
      deadline_millis=1000UL,
    ),
  )
  let second_prepared = finish_foundation_pool_work(pool, second_work)
  let retained_second = second_prepared
  assert_eq(pool.free_lane_count(), 0)
  let first = begin_foundation_session(instance, first_prepared)
  assert_pool_work_stale(first_work)
  assert_prepared_shell_revoked(retained_first)
  assert_eq(instance.begin(second_prepared).kind(), LunaOnlineInstanceBusy)
  assert_eq(second_prepared.input_token_count(), 2)
  let queued_expired = prepare_foundation_request(
    tokenizer,
    session_received(
      Streaming,
      request_id=953UL,
      max_new_tokens=1,
      deadline_millis=100UL,
    ),
  )
  assert_eq(instance.begin(queued_expired).kind(), LunaOnlineInstanceBusy)
  finish_persistent_natural(first)
  assert_eq(pool.free_lane_count(), 1)

  let clock = @monotonic_clock.MonotonicClock::system()
  let expires_after = clock.now_millis() + 101UL
  while clock.now_millis() < expires_after {

  }
  try instance.begin(queued_expired) catch {
    LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure) => ()
    _ => raise Failure("unexpected queued-expiry admission error")
  } noraise {
    _ => raise Failure("expired prepared request entered the scheduler")
  }

  let second = begin_foundation_session(instance, second_prepared)
  assert_prepared_shell_revoked(retained_second)
  assert_false(first.ticket == second.ticket)
  assert_stale_online_ticket(instance, first.ticket)
  finish_persistent_cancelled(second)
  assert_eq(pool.free_lane_count(), 2)

  try instance.begin(second_prepared) catch {
    LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure) => ()
    _ => raise Failure("unexpected prepared-request replay error")
  } noraise {
    _ => raise Failure("one prepared request was admitted twice")
  }

  let third_work = submit_foundation_pool_work(
    pool,
    session_received(
      Streaming,
      request_id=954UL,
      max_new_tokens=1,
      deadline_millis=1000UL,
    ),
  )
  let third_prepared = finish_foundation_pool_work(pool, third_work)
  let third = begin_foundation_session(instance, third_prepared)
  finish_persistent_natural(third)
  assert_eq(pool.free_lane_count(), 2)

  // Token 2 passes request preparation, but exceeds this worker's token-id
  // capacity. Lower admission must reject after claim transfer and return the
  // exact preparation lane.
  let lower_instance = prepare_lower_reject_foundation_instance(
    executable,
    tokenizer.digest(),
  )
  let lower_rejected = prepare_foundation_pool_request(
    pool,
    session_received(Streaming, request_id=955UL, max_new_tokens=1),
  )
  assert_eq(pool.free_lane_count(), 1)
  try lower_instance.begin(lower_rejected) catch {
    LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure) => ()
    _ => raise Failure("unexpected lower capacity-admission rollback error")
  } noraise {
    _ => raise Failure("out-of-range token bypassed lower scheduler capacity")
  }
  assert_eq(pool.free_lane_count(), 2)
  assert_prepared_shell_revoked(lower_rejected)
  drain_foundation_instance(lower_instance)

  let draining_prepared = prepare_foundation_pool_request(
    pool,
    session_received(Streaming, request_id=952UL),
  )
  assert_eq(pool.free_lane_count(), 1)
  instance.begin_drain()
  assert_eq(
    instance.begin(draining_prepared).kind(),
    LunaOnlineInstanceDraining,
  )
  assert_eq(draining_prepared.input_token_count(), 2)
  let retry_instance = prepare_foundation_instance(
    executable,
    tokenizer.digest(),
  )
  let draining_retry = begin_foundation_session(
    retry_instance, draining_prepared,
  )
  draining_retry.ack_event()
  draining_retry.begin_abort()
  progress_foundation_to_idle(draining_retry)
  assert_eq(pool.free_lane_count(), 2)
  drain_foundation_instance(retry_instance)
  assert_eq(instance.progress_shutdown(), LunaOnlineInstanceShutdownClosed)
  assert_eq(instance.progress_shutdown(), LunaOnlineInstanceShutdownClosed)
}
