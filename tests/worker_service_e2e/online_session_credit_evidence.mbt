///|
fn assert_online_credit_alias_rejected(
  credit : @online_session.LunaOnlineEventCredit,
) -> Unit raise {
  try ignore(credit.view()) catch {
    LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit)) => ()
    _ => raise Failure("unexpected stale Luna event credit error")
  } noraise {
    _ => raise Failure("stale Luna event credit retained read authority")
  }
  try credit.ack() catch {
    LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit)) => ()
    _ => raise Failure("unexpected stale Luna event ACK error")
  } noraise {
    _ => raise Failure("stale Luna event credit retained ACK authority")
  }
}

///|
fn assert_online_aborted_credit_rejected(
  credit : @online_session.LunaOnlineEventCredit,
) -> Unit raise {
  try ignore(credit.view()) catch {
    LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineLifecycle)) => ()
    _ => raise Failure("unexpected aborted Luna event credit error")
  } noraise {
    _ => raise Failure("aborted Luna event credit retained read authority")
  }
  try credit.ack() catch {
    LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineLifecycle)) => ()
    _ => raise Failure("unexpected aborted Luna event ACK error")
  } noraise {
    _ => raise Failure("aborted Luna event credit retained ACK authority")
  }
}

///|
fn run_online_session_credit_evidence(executable : Bytes) -> Unit raise {
  let tokenizer = session_tokenizer()
  let instance = prepare_foundation_instance(executable, tokenizer.digest())
  let preflight_prepared = prepare_foundation_request(
    tokenizer,
    session_received(Streaming, request_id=918UL, max_new_tokens=1),
  )
  try
    @framed_wire.LunaFramedEventAdapter::new(
      @framed_wire.FramedWireLimits::new(
        inference_limits(),
        max_frame_bytes=255,
      ),
    )
  catch {
    @framed_wire.InvalidLimit(FrameBytes) => ()
    _ => raise Failure("unexpected Luna adapter preflight error")
  } noraise {
    _ => raise Failure("undersized Luna adapter passed preflight")
  }
  let preflight_retry = begin_foundation_session(instance, preflight_prepared)
  preflight_retry.ack_event()
  preflight_retry.begin_abort()
  progress_foundation_to_idle(preflight_retry)

  let session = begin_foundation_session(
    instance,
    prepare_foundation_request(
      tokenizer,
      session_received(Streaming, request_id=916UL, max_new_tokens=1),
    ),
  )
  let accepted_credit = instance.take_event(session.ticket)
  let accepted_alias = accepted_credit
  let accepted = accepted_credit.view().accepted()
  assert_eq(accepted.model_identity(), identity())
  try ignore(instance.take_event(session.ticket)) catch {
    LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit)) => ()
    _ => raise Failure("unexpected double Luna event acquisition error")
  } noraise {
    _ => raise Failure("one semantic publication issued two event credits")
  }
  accepted_credit.ack()
  assert_online_credit_alias_rejected(accepted_alias)

  progress_foundation_to_event(session)
  let token_credit = instance.take_event(session.ticket)
  let token = token_credit.view().token()
  assert_eq(token.token_id(), 0)
  assert_eq(token.position(), 0)
  token_credit.ack()

  progress_foundation_to_event(session)
  let usage_credit = instance.take_event(session.ticket)
  let usage_alias = usage_credit
  let usage = usage_credit.view().usage()
  assert_eq(usage.input_tokens(), 2)
  assert_eq(usage.output_tokens(), 1)
  usage_credit.ack()
  let completed_credit = instance.take_event(session.ticket)
  assert_online_credit_alias_rejected(usage_alias)
  let completed = completed_credit.view().completed()
  assert_eq(completed.finish_reason(), TokenLimit)
  assert_eq(completed.output_tokens(), 1)
  completed_credit.ack()
  progress_foundation_to_idle(session)

  let aborting = begin_foundation_session(
    instance,
    prepare_foundation_request(
      tokenizer,
      session_received(Streaming, request_id=917UL, max_new_tokens=1),
    ),
  )
  let abort_credit = instance.take_event(aborting.ticket)
  instance.begin_abort(aborting.ticket)
  assert_online_aborted_credit_rejected(abort_credit)
  progress_foundation_to_idle(aborting)
  drain_foundation_instance(instance)
}
