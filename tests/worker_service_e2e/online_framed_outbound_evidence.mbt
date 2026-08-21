///|
fn outbound_framed_limits() -> @framed_wire.FramedWireLimits raise {
  @framed_wire.FramedWireLimits::new(inference_limits(), max_frame_bytes=4096)
}

///|
fn assert_outbound_bytes_equal(
  actual : FixedArray[Byte],
  expected : FixedArray[Byte],
) -> Unit raise {
  assert_eq(actual.length(), expected.length())
  for index in 0..<actual.length() {
    assert_eq(actual[index], expected[index])
  }
}

///|
fn canonical_outbound_bytes(
  event : @inference.StreamEvent,
) -> FixedArray[Byte] raise {
  let frame = @framed_wire.EventFrameBuffer::new(outbound_framed_limits()).encode(
    event,
  )
  let output = FixedArray::make(frame.length(), b'\x00')
  ignore(frame.copy_to(output, destination_offset=0))
  output
}

///|
fn assert_cooperative_outbound_golden(
  semantic : @luna_event.LunaEventView,
  expected : @inference.StreamEvent,
) -> Unit raise {
  let golden = canonical_outbound_bytes(expected)
  let mut exact_total = 0UL
  for budget_units in [1, 17] {
    let workspace = @framed_wire.LunaFramedEventWorkspace::new(
      outbound_framed_limits(),
      @framed_wire.LunaFramedEventStepBudget::new(budget_units),
    )
    let work = workspace.begin(semantic)
    let mut summed = 0UL
    while work.state() == LunaFramedEventPending {
      assert_eq(work.progress(), LunaFramedEventAdvanced)
      let used = work.last_work_units()
      assert_true(used > 0)
      assert_true(used <= budget_units)
      summed += used.to_uint64()
    }
    assert_eq(work.total_work_units(), summed)
    assert_eq(work.progress(), LunaFramedEventIdle)
    assert_eq(work.last_work_units(), 0)
    if exact_total == 0UL {
      exact_total = summed
    } else {
      assert_eq(summed, exact_total)
    }
    let view = work.take_view()
    let actual = FixedArray::make(view.length(), b'\x00')
    let mut cursor = 0
    while cursor < actual.length() {
      let amount = Int::min(budget_units, actual.length() - cursor)
      assert_eq(
        view.copy_chunk_to(
          actual,
          destination_offset=cursor,
          source_offset=cursor,
          length=amount,
        ),
        amount,
      )
      cursor += amount
    }
    assert_outbound_bytes_equal(actual, golden)
    view.release()
  }
}

///|
fn run_cooperative_outbound_goldens() -> Unit raise {
  let inference = inference_limits()
  let owner = @luna_event.LunaEventOwner::new(inference)
  let effective = @inference.EffectiveLimits::new(
    max_new_tokens=2,
    context_ceiling=4,
    inference,
  )
  owner.publish_accepted(identity(), effective)
  let accepted = owner.view()
  assert_cooperative_outbound_golden(
    accepted,
    Accepted(
      @inference.AcceptedEvent::new(
        model_identity=identity(),
        effective_limits=effective,
      ),
    ),
  )
  owner.retire(accepted)

  let token_bytes : FixedArray[Byte] = [b'o', b'k']
  assert_eq(
    owner.publish_token_status(
      token_id=9,
      position=1,
      token_bytes,
      source_offset=0,
      length=2,
    ),
    LunaEventPublishReady,
  )
  let token = owner.view()
  assert_cooperative_outbound_golden(
    token,
    Token(
      @inference.TokenEvent::new(
        token_id=9,
        decoded_delta=Some(@inference.DecodedDelta::from_utf8(b"ok", inference)),
        position=1,
        inference,
      ),
    ),
  )
  owner.retire(token)

  owner.publish_usage(input_tokens=2, cached_input_tokens=1, output_tokens=1)
  let usage_view = owner.view()
  let usage = @inference.Usage::initial(
    input_tokens=2,
    cached_input_tokens=1,
    inference,
  ).advance(output_tokens=1, inference)
  assert_cooperative_outbound_golden(usage_view, Usage(usage))
  owner.retire(usage_view)

  let tail : FixedArray[Byte] = [b'e', b'n', b'd']
  let reasons : Array[@inference.FinishReason] = [
    StopSequence,
    TokenLimit,
    Cancelled,
  ]
  for reason in reasons {
    let tail_length = if reason == TokenLimit { tail.length() } else { 0 }
    owner.publish_completed(
      reason,
      input_tokens=2,
      cached_input_tokens=1,
      output_tokens=1,
      tail,
      tail_offset=0,
      tail_length~,
    )
    let completed = owner.view()
    let expected = if tail_length == 0 {
      @inference.CompletedEvent::new(finish_reason=reason, final_usage=usage)
    } else {
      @inference.CompletedEvent::with_final_delta(
        finish_reason=reason,
        final_usage=usage,
        final_delta=@inference.DecodedDelta::from_utf8(b"end", inference),
        inference,
      )
    }
    assert_cooperative_outbound_golden(completed, Completed(expected))
    owner.retire(completed)
  }

  let code : FixedArray[Byte] = [b'o', b'u', b't', b'p', b'u', b't']
  for retryable in [false, true] {
    owner.publish_failed(
      code,
      code_offset=0,
      code_length=code.length(),
      retryable~,
    )
    let failed = owner.view()
    assert_cooperative_outbound_golden(
      failed,
      Failed(
        @inference.FailedEvent::new(
          code=@inference.PublicErrorCode::from_ascii(b"output"),
          retryable~,
        ),
      ),
    )
    owner.retire(failed)
  }
}

///|
fn assert_online_credit_pins_progress(
  instance : @online_session.LunaOnlineInstance,
  ticket : @online_session.LunaOnlineRequestTicket,
) -> Unit raise {
  try ignore(instance.progress(ticket)) catch {
    LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit)) => ()
    _ => raise Failure("unexpected pinned outbound event error")
  } noraise {
    _ => raise Failure("framed outbound work acknowledged semantic credit")
  }
}

///|
fn run_online_framed_outbound_evidence(executable : Bytes) -> Unit raise {
  run_cooperative_outbound_goldens()
  let tokenizer = session_tokenizer()
  let instance = prepare_foundation_instance(executable, tokenizer.digest())
  let session = begin_foundation_session(
    instance,
    prepare_foundation_request(
      tokenizer,
      session_received(Streaming, request_id=940UL, max_new_tokens=1),
    ),
  )
  let credit = instance.take_event(session.ticket)
  let compatibility = @framed_wire.LunaFramedEventAdapter::new(
    outbound_framed_limits(),
  )
  compatibility.stage(credit.view())
  let golden = FixedArray::make(compatibility.length(), b'\x00')
  ignore(compatibility.copy_to(golden, destination_offset=0))
  compatibility.release()

  let workspace = @framed_wire.LunaFramedEventWorkspace::new(
    outbound_framed_limits(),
    @framed_wire.LunaFramedEventStepBudget::new(1),
  )
  let work = workspace.begin(credit.view())
  assert_eq(work.progress(), LunaFramedEventAdvanced)
  assert_online_credit_pins_progress(instance, session.ticket)
  while work.state() == LunaFramedEventPending {
    ignore(work.progress())
    assert_true(work.last_work_units() <= 1)
  }
  let frame = work.take_view()
  let actual = FixedArray::make(frame.length(), b'\x00')
  for cursor in 0..<actual.length() {
    ignore(
      frame.copy_chunk_to(
        actual,
        destination_offset=cursor,
        source_offset=cursor,
        length=1,
      ),
    )
    if cursor == 0 || cursor + 1 == actual.length() {
      assert_online_credit_pins_progress(instance, session.ticket)
    }
  }
  assert_outbound_bytes_equal(actual, golden)
  frame.release()
  assert_online_credit_pins_progress(instance, session.ticket)
  credit.ack()

  session.begin_abort()
  progress_foundation_to_idle(session)

  let stale_session = begin_foundation_session(
    instance,
    prepare_foundation_request(
      tokenizer,
      session_received(Streaming, request_id=941UL, max_new_tokens=1),
    ),
  )
  let stale_credit = instance.take_event(stale_session.ticket)
  let stale_workspace = @framed_wire.LunaFramedEventWorkspace::new(
    outbound_framed_limits(),
    @framed_wire.LunaFramedEventStepBudget::new(1),
  )
  let stale_work = stale_workspace.begin(stale_credit.view())
  ignore(stale_work.progress())
  stale_session.begin_abort()
  ignore(stale_work.progress())
  assert_eq(stale_work.state(), LunaFramedEventFailed)
  assert_eq(stale_work.failure(), StaleFrame)
  stale_work.abort()
  assert_online_aborted_credit_rejected(stale_credit)
  progress_foundation_to_idle(stale_session)
  drain_foundation_instance(instance)
}
