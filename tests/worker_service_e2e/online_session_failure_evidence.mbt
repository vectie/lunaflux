///|
fn finish_online_failure_cleanup(session : FoundationSession) -> Unit raise {
  let mut attempts = 0
  while attempts < 1000000 {
    match session.progress_retirement() {
      LunaOnlineInstanceClosed => return
      LunaOnlineInstanceRetirementPending => ()
      LunaOnlineInstanceIdle =>
        raise Failure("worker failure recycled poisoned instance authority")
    }
    attempts += 1
  }
  raise Failure("online failure cleanup did not close")
}

///|
fn assert_worker_unavailable_bundle(session : FoundationSession) -> Unit raise {
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
  finish_online_failure_cleanup(session)
}

///|
fn run_online_session_progress_failure(executable : Bytes) -> Unit raise {
  let tokenizer = session_tokenizer()
  let recovery_pool = prepare_foundation_pool(tokenizer, 1)
  let instance = prepare_default_foundation_instance(executable)
  let recovery_work = submit_foundation_pool_work(
    recovery_pool,
    session_received(Streaming, request_id=906UL),
  )
  let session = begin_foundation_session(
    instance,
    finish_foundation_pool_work(recovery_pool, recovery_work),
  )
  assert_eq(recovery_pool.free_lane_count(), 0)
  session.ack_event()
  progress_foundation_to_terminalization(session)
  assert_eq(session.progress(), LunaOnlineInstanceTerminalizationRequired)
  assert_worker_unavailable_bundle(session)
  assert_eq(recovery_pool.free_lane_count(), 1)
  let recovery_reuse = prepare_foundation_pool_request(
    recovery_pool,
    session_received(Streaming, request_id=926UL),
  )
  assert_pool_work_stale(recovery_work)
  recovery_reuse.discard()
  assert_eq(recovery_pool.free_lane_count(), 1)
  assert_eq(
    instance
    .begin(
      prepare_foundation_request(
        session_tokenizer(),
        session_received(Streaming, request_id=916UL),
      ),
    )
    .kind(),
    LunaOnlineInstanceDraining,
  )

  let cancelled_instance = prepare_default_foundation_instance(executable)
  let cancelled = begin_default_foundation_session(
    cancelled_instance,
    session_received(Streaming, request_id=912UL),
  )
  cancelled.ack_event()
  assert_eq(cancelled.progress(), LunaOnlineInstancePending)
  cancelled.request_cancel()
  progress_foundation_to_terminalization(cancelled)
  assert_eq(cancelled.progress(), LunaOnlineInstanceTerminalizationRequired)
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
  finish_online_failure_cleanup(cancelled)

  let deadline_instance = prepare_default_foundation_instance(executable)
  let deadline = begin_default_foundation_session(
    deadline_instance,
    session_received(Streaming, request_id=913UL),
  )
  deadline.ack_event()
  assert_eq(deadline.progress(), LunaOnlineInstancePending)
  wait_past_online_deadline()
  assert_eq(deadline.progress(), LunaOnlineInstancePending)
  progress_foundation_to_terminalization(deadline)
  assert_eq(deadline.progress(), LunaOnlineInstanceTerminalizationRequired)
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
  finish_online_failure_cleanup(deadline)

  let aborting_instance = prepare_default_foundation_instance(executable)
  let aborting = begin_default_foundation_session(
    aborting_instance,
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
