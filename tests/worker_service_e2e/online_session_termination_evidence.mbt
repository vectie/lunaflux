///|
fn assert_cancel_bundle(
  session : FoundationSession,
  expected_output_tokens : Int,
) -> Unit raise {
  progress_foundation_to_event(session)
  match foundation_event(session) {
    Usage(value) => assert_eq(value.output_tokens(), expected_output_tokens)
    _ => raise Failure("caller cancellation did not publish Usage")
  }
  session.request_cancel()
  session.check_deadline()
  session.ack_event()
  match foundation_event(session) {
    Completed(value) => {
      assert_eq(value.finish_reason(), Cancelled)
      assert_eq(value.final_usage().output_tokens(), expected_output_tokens)
      assert_true(value.final_delta() is None)
    }
    _ => raise Failure("caller cancellation did not publish Completed")
  }
  session.ack_event()
  progress_foundation_to_idle(session)
}

///|
fn wait_past_online_deadline() -> Unit raise {
  let clock = @monotonic_clock.MonotonicClock::system()
  let target = clock.now_millis() + 1001UL
  let mut observed = clock.now_millis()
  while observed < target {
    observed = clock.now_millis()
  }
}

///|
fn assert_deadline_abort(
  instance : @online_session.LunaOnlineInstance,
  request_id : UInt64,
  at_failed : Bool,
) -> Unit raise {
  let session = begin_default_foundation_session(
    instance,
    session_received(Streaming, request_id~),
  )
  session.ack_event()
  assert_eq(session.progress(), LunaOnlineInstancePending)
  wait_past_online_deadline()
  progress_foundation_to_event(session)
  if at_failed {
    session.ack_event()
  }
  session.begin_abort()
  progress_foundation_to_idle(session)
}

///|
fn run_online_session_termination_evidence(executable : Bytes) -> Unit raise {
  let instance = prepare_default_foundation_instance(executable)
  let accepted_cancel = begin_default_foundation_session(
    instance,
    session_received(Streaming, request_id=920UL),
  )
  accepted_cancel.request_cancel()
  accepted_cancel.check_deadline()
  accepted_cancel.ack_event()
  wait_past_online_deadline()
  accepted_cancel.check_deadline()
  accepted_cancel.request_cancel()
  assert_cancel_bundle(accepted_cancel, 0)

  let live_cancel = begin_default_foundation_session(
    instance,
    session_received(Streaming, request_id=927UL),
  )
  live_cancel.ack_event()
  live_cancel.request_cancel()
  assert_cancel_bundle(live_cancel, 0)

  let token_cancel = begin_default_foundation_session(
    instance,
    session_received(Streaming, request_id=921UL),
  )
  token_cancel.ack_event()
  progress_foundation_to_event(token_cancel)
  match foundation_event(token_cancel) {
    Token(_) => ()
    _ => raise Failure("caller cancellation fixture did not pin Token")
  }
  token_cancel.request_cancel()
  token_cancel.check_deadline()
  token_cancel.ack_event()
  assert_cancel_bundle(token_cancel, 1)

  let flight_cancel = begin_default_foundation_session(
    instance,
    session_received(Streaming, request_id=926UL),
  )
  flight_cancel.ack_event()
  assert_eq(flight_cancel.progress(), LunaOnlineInstancePending)
  flight_cancel.request_cancel()
  assert_cancel_bundle(flight_cancel, 0)

  let expired = begin_default_foundation_session(
    instance,
    session_received(Streaming, request_id=922UL),
  )
  expired.check_deadline()
  expired.ack_event()
  assert_eq(expired.progress(), LunaOnlineInstancePending)
  wait_past_online_deadline()
  progress_foundation_to_event(expired)
  match foundation_event(expired) {
    Usage(value) => assert_eq(value.output_tokens(), 0)
    _ => raise Failure("deadline did not publish Usage")
  }
  expired.request_cancel()
  expired.check_deadline()
  expired.ack_event()
  match foundation_event(expired) {
    Failed(value) => {
      assert_eq(value.code().as_string(), "deadline_exceeded")
      assert_false(value.retryable())
    }
    _ => raise Failure("deadline did not publish canonical Failed")
  }
  expired.ack_event()
  progress_foundation_to_idle(expired)
  assert_deadline_abort(instance, 924UL, false)
  assert_deadline_abort(instance, 925UL, true)

  let overdue_cancel = begin_default_foundation_session(
    instance,
    session_received(Streaming, request_id=928UL),
  )
  overdue_cancel.ack_event()
  wait_past_online_deadline()
  overdue_cancel.request_cancel()
  progress_foundation_to_event(overdue_cancel)
  match foundation_event(overdue_cancel) {
    Usage(_) => ()
    _ => raise Failure("overdue caller cancellation overrode deadline")
  }
  overdue_cancel.ack_event()
  match foundation_event(overdue_cancel) {
    Failed(value) => assert_eq(value.code().as_string(), "deadline_exceeded")
    _ => raise Failure("overdue cancellation did not retain deadline Failed")
  }
  overdue_cancel.ack_event()
  progress_foundation_to_idle(overdue_cancel)

  let natural = begin_default_foundation_session(
    instance,
    session_received(Streaming, request_id=923UL, max_new_tokens=1),
  )
  natural.ack_event()
  progress_foundation_to_event(natural)
  wait_past_online_deadline()
  natural.request_cancel()
  natural.check_deadline()
  match foundation_event(natural) {
    Token(_) => ()
    _ => raise Failure("natural terminal did not retain Token precedence")
  }
  natural.ack_event()
  progress_foundation_to_event(natural)
  match foundation_event(natural) {
    Usage(value) => assert_eq(value.output_tokens(), 1)
    _ => raise Failure("natural terminal did not retain Usage precedence")
  }
  natural.ack_event()
  match foundation_event(natural) {
    Completed(value) => assert_eq(value.finish_reason(), TokenLimit)
    _ => raise Failure("natural terminal did not beat late controls")
  }
  natural.ack_event()
  progress_foundation_to_idle(natural)
  drain_foundation_instance(instance)
}
