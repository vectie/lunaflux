///|
fn finish_online_failure_cleanup(
  session : @online_session.OnlineSession,
) -> Unit raise {
  let mut attempts = 0
  while attempts < 1000000 {
    match session.progress_cleanup() {
      OnlineSessionCleanupClosed => return
      OnlineSessionCleanupPending => ()
    }
    attempts += 1
  }
  raise Failure("online failure cleanup did not close")
}

///|
fn assert_worker_unavailable_bundle(
  session : @online_session.OnlineSession,
) -> Unit raise {
  progress_foundation_to_event(session)
  match foundation_event(session) {
    Usage(value) => assert_eq(value.output_tokens(), 0)
    _ => raise Failure("worker loss did not publish Usage")
  }
  session.request_cancel()
  session.check_deadline()
  session.ack_event()
  match foundation_event(session) {
    Failed(value) => {
      assert_eq(value.code().as_string(), "worker_unavailable")
      assert_true(value.retryable())
    }
    _ => raise Failure("worker loss did not publish canonical Failed")
  }
  session.ack_event()
  assert_eq(session.progress_cleanup(), OnlineSessionCleanupClosed)
}

///|
fn run_online_session_progress_failure(executable : Bytes) -> Unit raise {
  let session = prepare_foundation_session(
    executable,
    session_tokenizer(),
    session_received(Streaming, request_id=906UL),
  )
  session.ack_event()
  progress_foundation_to_terminalization(session)
  assert_eq(session.progress(), OnlineSessionTerminalizationRequired)
  assert_worker_unavailable_bundle(session)

  let cancelled = prepare_foundation_session(
    executable,
    session_tokenizer(),
    session_received(Streaming, request_id=912UL),
  )
  cancelled.ack_event()
  assert_eq(cancelled.progress(), OnlineSessionPending)
  cancelled.request_cancel()
  progress_foundation_to_terminalization(cancelled)
  assert_eq(cancelled.progress(), OnlineSessionTerminalizationRequired)
  progress_foundation_to_event(cancelled)
  match foundation_event(cancelled) {
    Usage(value) => assert_eq(value.output_tokens(), 0)
    _ => raise Failure("worker-loss caller cut did not publish Usage")
  }
  cancelled.ack_event()
  match foundation_event(cancelled) {
    Completed(value) => {
      assert_eq(value.finish_reason(), Cancelled)
      assert_true(value.final_delta() is None)
    }
    _ => raise Failure("worker loss overwrote committed caller cut")
  }
  cancelled.ack_event()
  assert_eq(cancelled.progress_cleanup(), OnlineSessionCleanupClosed)

  let deadline = prepare_foundation_session(
    executable,
    session_tokenizer(),
    session_received(Streaming, request_id=913UL),
  )
  deadline.ack_event()
  assert_eq(deadline.progress(), OnlineSessionPending)
  wait_past_online_deadline()
  assert_eq(deadline.progress(), OnlineSessionPending)
  progress_foundation_to_terminalization(deadline)
  assert_eq(deadline.progress(), OnlineSessionTerminalizationRequired)
  progress_foundation_to_event(deadline)
  match foundation_event(deadline) {
    Usage(value) => assert_eq(value.output_tokens(), 0)
    _ => raise Failure("worker-loss deadline cut did not publish Usage")
  }
  deadline.ack_event()
  match foundation_event(deadline) {
    Failed(value) => {
      assert_eq(value.code().as_string(), "deadline_exceeded")
      assert_false(value.retryable())
    }
    _ => raise Failure("worker loss overwrote committed deadline cut")
  }
  deadline.ack_event()
  assert_eq(deadline.progress_cleanup(), OnlineSessionCleanupClosed)

  let aborting = prepare_foundation_session(
    executable,
    session_tokenizer(),
    session_received(Streaming, request_id=914UL),
  )
  aborting.ack_event()
  progress_foundation_to_terminalization(aborting)
  progress_foundation_to_event(aborting)
  match foundation_event(aborting) {
    Usage(_) => ()
    _ => raise Failure("worker recovery abort did not pin Usage")
  }
  aborting.begin_abort()
  finish_online_failure_cleanup(aborting)
}
