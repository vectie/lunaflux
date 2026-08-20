///|
fn assert_stale_online_ticket(
  instance : @online_session.LunaOnlineInstance,
  stale : @online_session.LunaOnlineRequestTicket,
) -> Unit raise {
  try instance.has_event(stale) catch {
    LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineRequestEpoch)) => ()
    _ => raise Failure("unexpected stale online request ticket error")
  } noraise {
    _ => raise Failure("stale online request ticket retained authority")
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
  let instance = prepare_foundation_instance(executable, session_tokenizer())
  let first = begin_foundation_session(
    instance,
    session_received(
      Streaming,
      request_id=950UL,
      max_new_tokens=1,
      deadline_millis=1000UL,
    ),
  )
  let second_received = session_received(
    Streaming,
    request_id=951UL,
    max_new_tokens=2,
    deadline_millis=1000UL,
  )
  assert_eq(instance.begin(second_received), LunaOnlineInstanceBusy)
  finish_persistent_natural(first)

  let second = begin_foundation_session(instance, second_received)
  assert_false(first.ticket == second.ticket)
  assert_stale_online_ticket(instance, first.ticket)
  finish_persistent_cancelled(second)

  instance.begin_drain()
  assert_eq(
    instance.begin(session_received(Streaming, request_id=952UL)),
    LunaOnlineInstanceDraining,
  )
  assert_eq(instance.progress_shutdown(), LunaOnlineInstanceShutdownClosed)
  assert_eq(instance.progress_shutdown(), LunaOnlineInstanceShutdownClosed)
}
