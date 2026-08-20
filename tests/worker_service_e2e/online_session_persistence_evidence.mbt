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
fn run_online_session_persistence_evidence(executable : Bytes) -> Unit raise {
  let tokenizer = session_tokenizer()
  let instance = prepare_foundation_instance(executable, tokenizer.digest())
  let first_prepared = prepare_foundation_request(
    tokenizer,
    session_received(
      Streaming,
      request_id=950UL,
      max_new_tokens=1,
      deadline_millis=1000UL,
    ),
  )
  let second_prepared = prepare_foundation_request(
    tokenizer,
    session_received(
      Streaming,
      request_id=951UL,
      max_new_tokens=2,
      deadline_millis=1000UL,
    ),
  )
  let retained_second = second_prepared
  let first = begin_foundation_session(instance, first_prepared)
  assert_eq(instance.begin(second_prepared).kind(), LunaOnlineInstanceBusy)
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

  try instance.begin(second_prepared) catch {
    LunaOnlineInstanceFailed(LunaOnlineAdmissionFailure) => ()
    _ => raise Failure("unexpected prepared-request replay error")
  } noraise {
    _ => raise Failure("one prepared request was admitted twice")
  }

  let third = begin_foundation_session(
    instance,
    prepare_foundation_request(
      tokenizer,
      session_received(
        Streaming,
        request_id=954UL,
        max_new_tokens=1,
        deadline_millis=1000UL,
      ),
    ),
  )
  finish_persistent_natural(third)

  let draining_prepared = prepare_foundation_request(
    tokenizer,
    session_received(Streaming, request_id=952UL),
  )
  instance.begin_drain()
  assert_eq(
    instance.begin(draining_prepared).kind(),
    LunaOnlineInstanceDraining,
  )
  assert_eq(
    draining_prepared.take_claim().scheduler_request().request_id().value(),
    952UL,
  )
  assert_eq(instance.progress_shutdown(), LunaOnlineInstanceShutdownClosed)
  assert_eq(instance.progress_shutdown(), LunaOnlineInstanceShutdownClosed)
}
